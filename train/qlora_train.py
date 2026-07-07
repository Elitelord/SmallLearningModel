"""Part C - QLoRA fine-tune of Qwen3-0.6B Instruct on the generated dataset.

TWO code paths, chosen by hardware:
  * CUDA present (Colab GPU, the REAL run): Unsloth 4-bit QLoRA + TRL SFTTrainer.
  * CPU only (this Windows box, the SMOKE run): plain `peft` LoRA in fp32 via a
    tiny transformers Trainer loop. bitsandbytes/Unsloth need CUDA, so 4-bit is
    impossible here - but the loop still exercises the full generate->train->eval
    plumbing (Part E), which is the whole point of Day 2.

Either way we:
  - format with the Qwen3 chat template (tokenizer.apply_chat_template),
  - use the MINIMAL prompt from data.sft_format (identical to inference),
  - train only on the assistant tokens (prompt tokens are masked to -100),
  - save the LoRA adapter under train/adapters/<name>.

Usage:
    # local CPU smoke (few steps, tiny)
    .venv\\Scripts\\python -m train.qlora_train --data data/generated_v0.jsonl \\
        --adapter-name smoke --max-steps 10
    # Colab GPU real run
    python -m train.qlora_train --data data/generated_v1.jsonl --adapter-name v1 --epochs 3
"""

import argparse
import json
import sys
from pathlib import Path

from data.sft_format import build_messages, build_training_messages

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
ADAPTER_DIR = Path(__file__).resolve().parent / "adapters"

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_records(path: str) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if not recs:
        raise SystemExit(f"No training records in {path}")
    return recs


# --------------------------------------------------------------------------- #
# GPU path: Unsloth 4-bit QLoRA + TRL SFTTrainer (Colab).                      #
# --------------------------------------------------------------------------- #
def train_unsloth(records, args):
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=LORA_TARGETS,
        use_gradient_checkpointing="unsloth",
    )

    def to_text(rec):
        msgs = build_training_messages(rec["concept"], rec["explanation"])
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    ds = Dataset.from_list([{"text": to_text(r)} for r in records])

    sft_args = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=min(5, len(records)),
        num_train_epochs=args.epochs if args.max_steps is None else 1,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=args.lr,
        logging_steps=1,
        output_dir=str(ADAPTER_DIR / f"_trainer_{args.adapter_name}"),
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=sft_args)
    trainer.train()

    out = ADAPTER_DIR / args.adapter_name
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\nSaved Unsloth QLoRA adapter -> {out}")


# --------------------------------------------------------------------------- #
# CPU path: plain peft LoRA in fp32, tiny transformers Trainer (smoke test).  #
# --------------------------------------------------------------------------- #
def train_cpu_peft(records, args):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    print("[CPU fallback] plain peft LoRA in fp32 (no 4-bit). Smoke test only.")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Build label-masked examples: only assistant tokens contribute to the loss.
    # Render to text with the chat template, then tokenize to plain int lists
    # (apply_chat_template(tokenize=True) can return an Encoding, not a list).
    examples = []
    for rec in records:
        full_text = tok.apply_chat_template(
            build_training_messages(rec["concept"], rec["explanation"]),
            tokenize=False, add_generation_prompt=False,
        )
        prompt_text = tok.apply_chat_template(
            build_messages(rec["concept"]), tokenize=False, add_generation_prompt=True,
        )
        full = tok(full_text, add_special_tokens=False)["input_ids"]
        prompt = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full = full[: args.max_seq_len]
        labels = list(full)
        for i in range(min(len(prompt), len(labels))):
            labels[i] = -100  # mask the prompt; train only on the answer
        examples.append({"input_ids": full, "labels": labels})

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        input_ids, attn, labels = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad] * n)
            attn.append([1] * len(b["input_ids"]) + [0] * n)
            labels.append(b["labels"] + [-100] * n)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(labels),
        }

    targs = TrainingArguments(
        output_dir=str(ADAPTER_DIR / f"_trainer_{args.adapter_name}"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=examples, data_collator=collate)
    trainer.train()

    out = ADAPTER_DIR / args.adapter_name
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"\nSaved CPU peft LoRA adapter -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", default=str(REPO / "data" / "generated_v0.jsonl"))
    ap.add_argument("--adapter-name", default="smoke")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke test)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--force-cpu", action="store_true", help="force the CPU peft path")
    args = ap.parse_args()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records(args.data)
    print(f"Loaded {len(records)} training records from {args.data}")

    import torch

    use_gpu = torch.cuda.is_available() and not args.force_cpu
    if use_gpu:
        print("CUDA detected -> Unsloth 4-bit QLoRA path")
        train_unsloth(records, args)
    else:
        print("No CUDA (or --force-cpu) -> CPU peft LoRA fallback")
        train_cpu_peft(records, args)


if __name__ == "__main__":
    main()
