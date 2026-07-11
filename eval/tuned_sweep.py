"""Generate a tuned-only temperature/seed sweep while loading the adapter once."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from eval.base_vs_tuned import generate
from litmus.fk_score import ARI_BAND_V4, DISPERSION_MAX_V4, WP_BAND_V4, score_text

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
CONCEPTS_PATH = REPO_ROOT / "data" / "concepts.json"
HELD_OUT_EVAL_KEYS = {"eval", "eval_litmus", "blind_v4r5"}


def parse_csv(value: str, cast, name: str) -> list:
    try:
        parsed = [cast(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid {name}: {value}") from error
    if not parsed:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return parsed


def setting_key(temperature: float, seed: int) -> str:
    temperature_text = str(temperature).replace(".", "p")
    return f"t{temperature_text}_s{seed}"


def expected_setting_count(temperatures: list[float], seeds: list[int]) -> int:
    return sum(1 if temperature == 0 else len(seeds) for temperature in temperatures)


def validate_eval_policy(
    eval_key: str,
    temperatures: list[float],
    seeds: list[int],
    top_settings: int,
    final_eval: bool,
) -> None:
    if eval_key not in HELD_OUT_EVAL_KEYS:
        if final_eval:
            raise ValueError("--final-eval is only valid for held-out eval keys")
        return
    if not final_eval:
        raise ValueError(
            "held-out prompts require --final-eval; use calibration_v4r5 to select settings"
        )
    if len(temperatures) != 1:
        raise ValueError("held-out evaluation requires one preselected temperature")
    required_settings = expected_setting_count(temperatures, seeds)
    if top_settings < required_settings:
        raise ValueError(
            "held-out evaluation cannot rank away seeds; --top-settings must keep every setting"
        )


def readability_penalty(score: dict) -> float:
    fk = score["whole_passage_fk"]
    ari = score["whole_passage_ari"]
    dispersion = score["fk_stdev"]
    return round(
        max(0.0, WP_BAND_V4[0] - fk)
        + max(0.0, fk - WP_BAND_V4[1])
        + max(0.0, ARI_BAND_V4[0] - ari)
        + max(0.0, ari - ARI_BAND_V4[1])
        + max(0.0, dispersion - DISPERSION_MAX_V4),
        4,
    )


def load_tuned_model(model_id: str, adapter: Path, max_seq_len: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization,
            device_map={"": 0},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    if hasattr(model, "config"):
        model.config.max_position_embeddings = max(
            getattr(model.config, "max_position_embeddings", max_seq_len), max_seq_len
        )
    return model, tokenizer


def normalized_record(adapter_name: str, model_id: str, setting: str, concept: str,
                      text: str, readability: dict, source: str) -> dict:
    model_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", f"{adapter_name}_{setting}")
    return {
        "model_key": model_key,
        "label": f"{adapter_name} ({setting})",
        "model_id": model_id,
        "tested_family": "qwen",
        "prompt": "bare Explain:",
        "concept": concept,
        "text": text,
        "readability_pass": bool(readability["readability_pass_v4"]),
        "source": source,
        "temperature_seed": setting,
        "readability": {
            "whole_passage_fk": readability["whole_passage_fk"],
            "whole_passage_ari": readability["whole_passage_ari"],
            "fk_stdev": readability["fk_stdev"],
            "max_fk": readability["max_fk"],
            "penalty": readability_penalty(readability),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--eval-key", default="eval_litmus")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperatures", default="0,0.3,0.7")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--top-settings", type=int, default=2)
    parser.add_argument(
        "--final-eval",
        action="store_true",
        help="confirm a frozen held-out run; forbids temperature selection and seed filtering",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    temperatures = parse_csv(args.temperatures, float, "temperatures")
    seeds = parse_csv(args.seeds, int, "seeds")
    if any(temperature < 0 for temperature in temperatures):
        parser.error("temperatures must be non-negative")
    if args.top_settings < 1:
        parser.error("--top-settings must be positive")
    if args.start < 0 or args.limit < 0:
        parser.error("--start and --limit must be non-negative")
    try:
        validate_eval_policy(
            args.eval_key,
            temperatures,
            seeds,
            args.top_settings,
            args.final_eval,
        )
    except ValueError as error:
        parser.error(str(error))

    concepts_data = json.loads(CONCEPTS_PATH.read_text(encoding="utf-8"))
    concepts = concepts_data[args.eval_key][args.start:]
    if args.limit:
        concepts = concepts[: args.limit]
    if not concepts:
        parser.error("the selected concept slice is empty")
    model, tokenizer = load_tuned_model(args.model, args.adapter, args.max_seq_len)

    import torch

    settings = []
    all_records = []
    for temperature in temperatures:
        active_seeds = seeds[:1] if temperature == 0 else seeds
        for seed in active_seeds:
            key = setting_key(temperature, seed)
            setting_records = []
            print(f"\n=== {key} ===", flush=True)
            for index, concept in enumerate(concepts):
                concept_seed = seed * 1000 + index
                torch.manual_seed(concept_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(concept_seed)
                text = generate(model, tokenizer, concept, args.max_new_tokens, temperature)
                readability = score_text(text)
                record = normalized_record(
                    args.adapter.name,
                    args.model,
                    key,
                    concept,
                    text,
                    readability,
                    args.out.name,
                )
                setting_records.append(record)
                all_records.append(record)
                print(
                    f"[{index + 1}/{len(concepts)}] "
                    f"read={record['readability_pass']} penalty={record['readability']['penalty']} "
                    f"{concept}",
                    flush=True,
                )
            settings.append(
                {
                    "key": key,
                    "temperature": temperature,
                    "seed": seed,
                    "readability_passes": sum(r["readability_pass"] for r in setting_records),
                    "mean_penalty": round(
                        sum(r["readability"]["penalty"] for r in setting_records)
                        / len(setting_records),
                        4,
                    ),
                }
            )

    ranked = sorted(
        settings,
        key=lambda setting: (
            -setting["readability_passes"],
            setting["mean_penalty"],
            setting["temperature"],
            setting["seed"],
        ),
    )
    selected_keys = {setting["key"] for setting in ranked[: args.top_settings]}
    selected_records = [
        record for record in all_records if record["temperature_seed"] in selected_keys
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "adapter": str(args.adapter),
                "model": args.model,
                "eval_key": args.eval_key,
                "settings": settings,
                "ranked_settings": ranked,
                "records": all_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    top_path = args.out.with_name(args.out.stem + ".top.json")
    top_path.write_text(
        json.dumps(
            {"selected_settings": ranked[: args.top_settings], "records": selected_records},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n| setting | readability | mean penalty |")
    print("|---|---:|---:|")
    for setting in ranked:
        print(
            f"| {setting['key']} | {setting['readability_passes']}/{len(concepts)} | "
            f"{setting['mean_penalty']} |"
        )
    print(f"\nwrote {args.out}")
    print(f"wrote {top_path}")


if __name__ == "__main__":
    main()
