"""Build a deterministic training-data audit without using benchmark failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from litmus.fk_score import score_text
from data.v4r3 import (
    TARGETED_V4R3_ITEMS,
    meets_target,
    norm_text,
    target_config,
    training_prompt,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "v4" / "gold_v4_r4.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "v4" / "v4r4_broad_accuracy_audit.json"
HISTORICAL_TARGETED_PROMPTS = {
    norm_text(training_prompt(record)) for record in TARGETED_V4R3_ITEMS
}


def reserved_prompt_keys() -> set[str]:
    concepts_path = REPO_ROOT / "data" / "concepts.json"
    concepts = json.loads(concepts_path.read_text(encoding="utf-8"))
    return {
        norm_text(prompt)
        for key in ("eval", "calibration_v4r5", "blind_v4r5")
        for prompt in concepts.get(key, [])
    }


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_rank(record: dict) -> str:
    value = f"{training_prompt(record)}\n{record.get('explanation', '')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def passes_v4r3_target(record: dict) -> bool:
    score = score_text(record.get("explanation", ""))
    return (
        "error" not in score
        and score.get("readability_pass_v4") is True
        and meets_target(score, target_config())
        and 4 <= score.get("n_sentences", 0) <= 6
    )


def select_records(
    records: list[dict], sample_size: int, require_v4r3_target: bool = False
) -> list[dict]:
    candidates = []
    seen_prompts = set()
    reserved = reserved_prompt_keys()
    for record in sorted(records, key=stable_rank):
        prompt_key = norm_text(training_prompt(record))
        concept_key = norm_text(record.get("concept", ""))
        if (
            prompt_key in seen_prompts
            or prompt_key in HISTORICAL_TARGETED_PROMPTS
            or prompt_key in reserved
            or concept_key in reserved
            or (require_v4r3_target and not passes_v4r3_target(record))
        ):
            continue
        seen_prompts.add(prompt_key)
        candidates.append(record)

    if sample_size == 0:
        return candidates

    selected = []
    selected_ids = set()
    seen_concepts = set()
    for record in candidates:
        concept_key = norm_text(record.get("concept", ""))
        if concept_key in seen_concepts:
            continue
        selected.append(record)
        selected_ids.add(id(record))
        seen_concepts.add(concept_key)
        if len(selected) >= sample_size:
            return selected

    for record in candidates:
        if id(record) in selected_ids:
            continue
        selected.append(record)
        if len(selected) >= sample_size:
            break
    return selected


def normalize(
    records: list[dict], source: Path, model_key: str, label: str
) -> list[dict]:
    normalized = []
    for record in records:
        metrics = record.get("metrics", record.get("fk", {}))
        teacher = record.get("teacher", "mixed")
        normalized.append(
            {
                "model_key": model_key,
                "label": label,
                "model_id": teacher,
                "tested_family": "mixed_teacher",
                "prompt": "training phrasing",
                "concept": training_prompt(record),
                "text": record["explanation"],
                "readability_pass": bool(
                    metrics.get("readability_pass_v4", True)
                ),
                "source": source.name,
                "training_metadata": {
                    "concept": record.get("concept"),
                    "teacher": teacher,
                    "rewriters": record.get("rewriters", []),
                    "judge": record.get("judge"),
                    "rewrite_iters": record.get("rewrite_iters", 0),
                    "accuracy_repairs": record.get("accuracy_repairs", 0),
                },
            }
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--model-key")
    parser.add_argument("--label")
    parser.add_argument("--require-v4r3-target", action="store_true")
    args = parser.parse_args()
    if args.sample_size < 0:
        parser.error("--sample-size must be non-negative (0 means all)")

    model_key = args.model_key or f"{args.input.stem}_training_audit"
    label = args.label or f"{args.input.stem} training-target audit"
    selected = select_records(
        load_jsonl(args.input), args.sample_size, args.require_v4r3_target
    )
    normalized = normalize(selected, args.input, model_key, label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": str(args.input),
                "selection_policy": (
                    "stable broad sample, one record per concept before repeats; "
                    "reserved and historical failure-targeted prompts excluded"
                ),
                "requested_records": args.sample_size,
                "require_v4r3_target": args.require_v4r3_target,
                "records": normalized,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(normalized)}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
