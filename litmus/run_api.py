"""Generate GPT outputs for all 12 concepts via the OpenAI API.

Writes litmus/outputs/gpt.json: {concept: output_text}. Also records the model
id and temperature used, for the limitations section.

Usage:
    .venv\\Scripts\\python -m litmus.run_api                 # default gpt-4o, temp 0.7
    .venv\\Scripts\\python -m litmus.run_api --model gpt-4.1 --temperature 0.7
"""

import argparse
import json
import sys
from pathlib import Path

from litmus.concepts import CONCEPTS, build_prompt
from litmus.env import load_env

# Windows consoles default to cp1252; model output may contain emoji/unicode.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--key", default="gpt", help="model key used in outputs/results")
    args = ap.parse_args()

    load_env()
    from openai import OpenAI

    client = OpenAI()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for i, concept in enumerate(CONCEPTS, 1):
        resp = client.chat.completions.create(
            model=args.model,
            temperature=args.temperature,
            messages=[{"role": "user", "content": build_prompt(concept)}],
        )
        text = resp.choices[0].message.content.strip()
        outputs[concept] = text
        print(f"[{i}/{len(CONCEPTS)}] {concept}")
        print(text)
        print("-" * 60)

    payload = {
        "model_key": args.key,
        "model_id": args.model,
        "temperature": args.temperature,
        "outputs": outputs,
    }
    out_path = OUT_DIR / f"{args.key}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
