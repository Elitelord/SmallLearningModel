"""Part E - the 50-junk end-to-end smoke test. THE Day-2 checkpoint.

Wires Part A (concepts + generate) -> Part C (train) -> Part D (eval) on ~50
throwaway examples. "Junk" data is fine here: the point is to prove the whole
loop completes without errors and prints a base-vs-tuned table, de-risking the
plumbing before spending on the real v1 dataset (Day 3).

This calls the same module entry points a real run uses, just with tiny counts
and the CPU training fallback.

Usage:
    .venv\\Scripts\\python -m scripts.smoke_e2e                 # full loop, uses API
    .venv\\Scripts\\python -m scripts.smoke_e2e --offline-concepts  # seed concepts, still calls teacher for gen
    .venv\\Scripts\\python -m scripts.smoke_e2e --no-judge      # skip API judge in eval
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(step, cmd):
    print(f"\n{'='*70}\n### {step}\n### {' '.join(cmd)}\n{'='*70}", flush=True)
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        print(f"\n!!! STEP FAILED: {step} (exit {r.returncode})")
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="junk examples to generate")
    ap.add_argument("--eval-n", type=int, default=5, help="held-out eval concepts")
    ap.add_argument("--max-steps", type=int, default=10, help="training steps (smoke)")
    ap.add_argument("--offline-concepts", action="store_true", help="seed concept list, no API")
    ap.add_argument("--no-judge", action="store_true", help="skip API judge in eval")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="eval generation length")
    ap.add_argument("--adapter-name", default="smoke")
    args = ap.parse_args()

    data_out = "data/generated_v0.jsonl"
    adapter = f"train/adapters/{args.adapter_name}"

    # A.1 - concepts
    concept_cmd = [PY, "-m", "data.gen_concepts", "--target", str(args.n + args.eval_n)]
    if args.offline_concepts:
        concept_cmd.append("--offline")
    run("A.1 concepts", concept_cmd)

    # A.3-5 - generate (junk mode, skip heavy filtering)
    run("A generate (junk)", [PY, "-m", "data.generate", "--junk",
                              "--limit", str(args.n), "--out", data_out])

    # C - train (CPU fallback, few steps)
    run("C train (CPU peft LoRA)", [PY, "-m", "train.qlora_train",
                                    "--data", data_out, "--adapter-name", args.adapter_name,
                                    "--max-steps", str(args.max_steps), "--force-cpu"])

    # D - eval (base vs tuned)
    eval_cmd = [PY, "-m", "eval.base_vs_tuned", "--adapter", adapter,
                "--limit", str(args.eval_n), "--max-new-tokens", str(args.max_new_tokens)]
    if args.no_judge:
        eval_cmd.append("--no-judge")
    run("D eval (base vs tuned)", eval_cmd)

    print("\n" + "="*70)
    print("SMOKE TEST COMPLETE: generate -> train -> eval ran end-to-end and printed")
    print("a base-vs-tuned table. Plumbing is de-risked for the Day-3 real dataset.")
    print("="*70)


if __name__ == "__main__":
    main()
