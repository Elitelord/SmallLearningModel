"""Part D - base vs tuned eval on the HELD-OUT concepts, reusing the litmus harness.

Runs base Qwen3-0.6B and the tuned (base + LoRA adapter) model on the eval-split
concepts with the SAME minimal prompt used in training (data.sft_format), then
scores both with:
  - the litmus FK harness (score_text): ceiling 3.0, band 2.0-3.0, % in band, and
  - the accuracy judge (0/1/2, mechanism rubric) - judge != student.

overall_pass = readability_pass AND accuracy == 2 (same as the litmus spec).

Prints a base-vs-tuned markdown table with per-concept rows and a delta summary,
and saves eval/base_vs_tuned_results.json.

Usage:
    # smoke: 5 concepts, judge on
    .venv\\Scripts\\python -m eval.base_vs_tuned --adapter train/adapters/smoke --limit 5
    # skip the API judge (FK only) for a pure-offline run
    .venv\\Scripts\\python -m eval.base_vs_tuned --adapter train/adapters/smoke --limit 5 --no-judge
"""

import argparse
import json
import re
import sys
from pathlib import Path

from data.sft_format import build_messages
from litmus.accuracy import build_judge_prompt
from litmus.fk_score import score_text

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
CONCEPTS_PATH = REPO / "data" / "concepts.json"
RESULTS_PATH = Path(__file__).resolve().parent / "base_vs_tuned_results.json"


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate(model, tok, concept, max_new_tokens, temperature):
    import torch

    prompt = tok.apply_chat_template(
        build_messages(concept),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.8 if temperature > 0 else None,
            pad_token_id=tok.eos_token_id,
        )
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return strip_think(gen)


def judge_accuracy(client, judge_model, concept, text):
    # audience-calibrated: judge mechanism at the grade-3 level (same ruler as
    # data-gen and the re-scored litmus baseline in litmus/results_v3.json).
    resp = client.chat.completions.create(
        model=judge_model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user",
                   "content": build_judge_prompt(concept, text, audience_calibrated=True)}],
    )
    data = json.loads(resp.choices[0].message.content)
    return int(data["score"]), data.get("justification", "")


def score_output(text, concept, judge_client, judge_model):
    fk = score_text(text)
    if "error" in fk:
        row = {"whole_passage_fk": None, "whole_passage_ari": None, "fk_stdev": None,
               "pct_in_band": None, "readability_pass": False, "accuracy": None,
               "overall_pass": False, "note": "no sentences"}
    else:
        # v4 gate is operative (whole-passage FK 3-6 AND ARI 3-7; matches data-gen
        # and the v4 dataset). max_fk/pct_in_band kept as diagnostics only.
        row = {
            "whole_passage_fk": fk["whole_passage_fk"],
            "whole_passage_ari": fk["whole_passage_ari"],
            "fk_stdev": fk["fk_stdev"],
            "max_fk": fk["max_fk"],
            "pct_in_band": fk["pct_in_band"],
            "readability_pass": fk["readability_pass_v4"],
        }
    if judge_client is not None and "error" not in fk:
        acc, note = judge_accuracy(judge_client, judge_model, concept, text)
        row["accuracy"] = acc
        row["overall_pass"] = bool(fk["readability_pass_v4"] and acc == 2)
        row["note"] = note
    else:
        row.setdefault("accuracy", None)
        row.setdefault("overall_pass", False)
        row.setdefault("note", "")
    return row


def run_model(model_id, adapter, concepts, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    on_gpu = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(model_id)
    if on_gpu:
        # 4-bit on the GPU: matches training precision, fits a T4, and — critically —
        # actually runs on the GPU. The old plain from_pretrained (no device_map / no
        # .to("cuda")) left the 4B model on CPU, so generation took HOURS on Colab.
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # T4 has no bf16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16,
        )
    else:
        # CPU fallback (local smoke test only).
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    outputs = {}
    for i, c in enumerate(concepts, 1):
        outputs[c] = generate(model, tok, c, args.max_new_tokens, args.temperature)
        print(f"  [{i}/{len(concepts)}] {c}", flush=True)  # flush: live progress in Colab
    del model
    if on_gpu:
        torch.cuda.empty_cache()
    return outputs


def fmt(v):
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")  # tune target upgraded 0.6B->4B
    ap.add_argument("--adapter", required=True, help="path to the tuned LoRA adapter")
    ap.add_argument("--judge", default="gpt-4o")
    ap.add_argument("--limit", type=int, default=5, help="how many eval concepts (smoke=5)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=350)
    ap.add_argument("--no-judge", action="store_true", help="FK only, no API accuracy judge")
    args = ap.parse_args()

    eval_concepts = json.loads(CONCEPTS_PATH.read_text(encoding="utf-8"))["eval"]
    if args.limit:
        eval_concepts = eval_concepts[: args.limit]

    judge_client = None
    if not args.no_judge:
        from litmus.env import load_env
        from openai import OpenAI

        load_env()
        judge_client = OpenAI()

    print(f"=== base vs tuned | {len(eval_concepts)} held-out concepts | "
          f"judge={'off' if args.no_judge else args.judge} ===\n")

    print("Generating BASE outputs...")
    base_out = run_model(args.model, None, eval_concepts, args)
    print("Generating TUNED outputs...")
    tuned_out = run_model(args.model, args.adapter, eval_concepts, args)

    print("\nScoring...")
    rows = []
    for c in eval_concepts:
        b = score_output(base_out[c], c, judge_client, args.judge)
        t = score_output(tuned_out[c], c, judge_client, args.judge)
        rows.append({"concept": c, "base": b, "tuned": t,
                     "base_text": base_out[c], "tuned_text": tuned_out[c]})

    # ---- table ----
    print("\n## Base vs Tuned (held-out concepts, identical minimal prompt)\n")
    header = ("| Concept | base wpFK | tuned wpFK | base std | tuned std | "
              "base acc | tuned acc | base pass | tuned pass |")
    print(header)
    print("|" + "---|" * 9)
    for r in rows:
        b, t = r["base"], r["tuned"]
        print(f"| {r['concept'][:34]} | {fmt(b['whole_passage_fk'])} | {fmt(t['whole_passage_fk'])} | "
              f"{fmt(b['fk_stdev'])} | {fmt(t['fk_stdev'])} | {fmt(b['accuracy'])} | "
              f"{fmt(t['accuracy'])} | {b['overall_pass']} | {t['overall_pass']} |")

    def agg(key, sub):
        vals = [r[key][sub] for r in rows if r[key][sub] is not None]
        return sum(vals) / len(vals) if vals else None

    def rate(key):
        return sum(1 for r in rows if r[key]["overall_pass"]) / len(rows)

    def read_rate(key):
        return sum(1 for r in rows if r[key]["readability_pass"]) / len(rows)

    print("\n## Deltas (tuned - base)\n")
    b_fk, t_fk = agg("base", "whole_passage_fk"), agg("tuned", "whole_passage_fk")
    b_ari, t_ari = agg("base", "whole_passage_ari"), agg("tuned", "whole_passage_ari")
    print("| metric | base | tuned | delta |")
    print("|---|---|---|---|")
    if b_fk is not None and t_fk is not None:
        print(f"| avg whole-passage FK (target 3.0-6.0) | {b_fk:.2f} | {t_fk:.2f} | {t_fk-b_fk:+.2f} |")
    if b_ari is not None and t_ari is not None:
        print(f"| avg whole-passage ARI (target 3.0-7.0) | {b_ari:.2f} | {t_ari:.2f} | {t_ari-b_ari:+.2f} |")
    print(f"| readability pass-rate | {read_rate('base'):.2f} | {read_rate('tuned'):.2f} | "
          f"{read_rate('tuned')-read_rate('base'):+.2f} |")
    if not args.no_judge:
        print(f"| overall pass-rate | {rate('base'):.2f} | {rate('tuned'):.2f} | "
              f"{rate('tuned')-rate('base'):+.2f} |")

    RESULTS_PATH.write_text(json.dumps({"rows": rows, "model": args.model,
                                        "adapter": args.adapter}, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    print("\nReadability pass = v4 gate (whole-passage FK 3-6 AND ARI 3-7, dispersion, "
          "backstop). overall pass = readability_pass AND accuracy==2. Run with --no-judge "
          "for an FK/ARI-only comparison (no API judge).")


if __name__ == "__main__":
    main()
