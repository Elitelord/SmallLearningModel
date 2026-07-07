"""Generate Qwen3 outputs locally on CPU for all 12 concepts.

Qwen3 is a hybrid think/instruct model; we disable thinking so it behaves like
the instruct target we would actually tune and serve. Any stray <think> block
is stripped defensively.

Writes litmus/outputs/<key>.json: {model_key, model_id, temperature, outputs}.

Usage:
    .venv\\Scripts\\python -m litmus.run_qwen                          # Qwen3-0.6B
    .venv\\Scripts\\python -m litmus.run_qwen --model Qwen/Qwen3-1.7B --key qwen_1.7b
"""

import argparse
import json
import re
import sys
from pathlib import Path

from litmus.concepts import CONCEPTS, build_prompt

# Windows consoles default to cp1252; model output contains emoji/unicode.
# Reconfigure so a print can't crash the run and lose generated work.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--key", default="qwen_0.6b")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=350)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for i, concept in enumerate(CONCEPTS, 1):
        messages = [{"role": "user", "content": build_prompt(concept)}]
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # instruct-style, no chain-of-thought
        )
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.8,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        text = strip_think(gen)
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
