"""Optional consistency probe — the 'every time' criterion.

Runs a model N times on a few concepts and reports run-to-run FK variance.
Prompting failing *on average* is good evidence; prompting being *inconsistent*
run-to-run is stronger, since the spec demands reliability every time.

Writes litmus/consistency.json and prints a summary.

Usage:
    .venv\\Scripts\\python -m litmus.consistency_probe --backend api --model gpt-4o --runs 3
    .venv\\Scripts\\python -m litmus.consistency_probe --backend qwen --model Qwen/Qwen3-0.6B --runs 3
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from litmus.concepts import build_prompt
from litmus.fk_score import score_text

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
PROBE_CONCEPTS = [
    "Why is the sky blue?",
    "How do magnets work?",
    "How do our lungs help us breathe?",
]


def gen_api(concepts, runs, model, temperature):
    from litmus.env import load_env

    load_env()
    from openai import OpenAI

    client = OpenAI()
    out = {c: [] for c in concepts}
    for c in concepts:
        for _ in range(runs):
            r = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": build_prompt(c)}],
            )
            out[c].append(r.choices[0].message.content.strip())
    return out


def gen_qwen(concepts, runs, model, temperature):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
    m.eval()
    out = {c: [] for c in concepts}
    for c in concepts:
        for run_i in range(runs):
            torch.manual_seed(run_i)  # vary seed per run
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": build_prompt(c)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inp = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                o = m.generate(
                    **inp,
                    max_new_tokens=350,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.8,
                    pad_token_id=tok.eos_token_id,
                )
            import re

            txt = tok.decode(o[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
            out[c].append(re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["api", "qwen"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--key", default=None)
    args = ap.parse_args()

    gen = gen_api if args.backend == "api" else gen_qwen
    outputs = gen(PROBE_CONCEPTS, args.runs, args.model, args.temperature)

    summary = {}
    for concept, texts in outputs.items():
        max_fks = [score_text(t)["max_fk"] for t in texts]
        reads = [score_text(t)["readability_pass"] for t in texts]
        summary[concept] = {
            "max_fk_per_run": max_fks,
            "max_fk_mean": round(statistics.mean(max_fks), 2),
            "max_fk_stdev": round(statistics.pstdev(max_fks), 2) if len(max_fks) > 1 else 0.0,
            "readability_pass_per_run": reads,
            "n_readability_pass": sum(reads),
            "runs": args.runs,
        }
        print(
            f"{concept}\n  max_fk per run: {max_fks} "
            f"(mean {summary[concept]['max_fk_mean']}, "
            f"stdev {summary[concept]['max_fk_stdev']}); "
            f"readability pass {sum(reads)}/{args.runs}"
        )

    key = args.key or f"{args.backend}_{args.model.split('/')[-1]}"
    payload = {"model": args.model, "backend": args.backend, "temperature": args.temperature, "summary": summary}
    out_path = HERE / f"consistency_{key}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
