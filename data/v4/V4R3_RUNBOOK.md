# v4r3 Readability Push Runbook

This loop keeps the v4 eval gate fixed and tightens only the training-data target.

## 1. Build the paid generation set

Use the existing r2 final file as a clean seed, dedupe by training prompt, exclude all
24 eval prompts, then generate until the final file reaches about 450 examples.

```bash
python -m data.generate \
  --seed data/v4/gold_v4_r2_final.jsonl \
  --target-kept 450 \
  --teachers <claude-compatible-slug>,<gemini-compatible-slug> \
  --rewriters <gemini-compatible-slug>,<claude-compatible-slug> \
  --judge <openai-compatible-judge-slug> \
  --out data/v4/gold_v4_r3.jsonl
```

The defaults are the v4r3 training target: FK 3.3-5.0, ARI 3.8-6.2, stdev <= 1.1,
max sentence FK <= 7.0, and 4-6 sentences.

## 2. Audit before training

```bash
python -m data.audit_v4r3 data/v4/gold_v4_r3.jsonl
```

The audit fails nonzero on duplicate training prompts, exact eval leakage, accuracy
scores other than 2, v4 readability failures, v4r3 target failures, or sentence-count
failures.

## 3. Train adapters

Main run:

```bash
python -m train.qlora_train \
  --data data/v4/gold_v4_r3.jsonl \
  --adapter-name v4r3 \
  --epochs 3 \
  --lora-r 16 --lora-alpha 32 \
  --batch-size 8 --grad-accum 2 --lr 2e-4
```

Cheap ablation:

```bash
python -m train.qlora_train \
  --data data/v4/gold_v4_r3.jsonl \
  --adapter-name v4r3_e2 \
  --epochs 2 \
  --lora-r 16 --lora-alpha 32 \
  --batch-size 8 --grad-accum 2 --lr 2e-4
```

## 4. Evaluate without overwriting r2

Original 12 litmus concepts:

```bash
python -m eval.base_vs_tuned \
  --adapter train/adapters/v4r3 \
  --eval-key eval_litmus \
  --limit 0 \
  --no-judge \
  --out eval/base_vs_tuned_v4r3_litmus12_readability.json
```

Full 24 held-out concepts:

```bash
python -m eval.base_vs_tuned \
  --adapter train/adapters/v4r3 \
  --eval-key eval \
  --limit 0 \
  --no-judge \
  --out eval/base_vs_tuned_v4r3_all24_readability.json
```

Repeat both without `--no-judge` once the gateway judge is configured.
