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

---

# v4r4 — mechanism-preserving push (A3 + A5 + B4)

Same v4 eval gate and same v4r3 training target (FK 3.3-5.0, ARI 3.8-6.2, stdev <= 1.1,
max sentence FK <= 7.0, 4-6 sentences). What changes:

- **A3 (mechanism-preserving generation):** the generation loop now satisfies readability
  AND accuracy *jointly*. A readable-but-inaccurate draft gets `--max-accuracy-repairs`
  (default 2) mechanism-repair rewrites — "restore the causal why, keep the same reading
  level" — instead of being judged only once at the end. Purpose: fix the v4r3 accuracy
  slip (11/12 -> 9/12) where aggressive readability rewrites thinned the mechanism.
- **A5:** scale to 600 kept.
- **B4:** raise LoRA capacity r16/a32 -> r32/a64.

## 1. Generate the v4r4 set (fresh, 600)

Fresh (no `--seed`) so every example goes through the A3 pipeline — clean attribution.

```bash
python -m data.generate \
  --target-kept 600 --concurrency 10 --max-accuracy-repairs 2 \
  --teachers claude-group/claude-sonnet-4-6,gemini-group/gemini-3.1-pro \
  --rewriters gemini-group/gemini-3.1-pro,claude-group/claude-sonnet-4-6 \
  --judge openai-group/gpt-4.1 --resume \
  --out data/v4/gold_v4_r4.jsonl
```

Fallback if the stricter joint gate misses 600 within the item pool: rerun the same
command with `--seed data/v4/gold_v4_r3.jsonl` added (top up from v4r3, `--resume` keeps
what's already written).

## 2. Audit (bands identical to v4r3, reuse as-is)

```bash
python -m data.audit_v4r3 data/v4/gold_v4_r4.jsonl
```

## 3. Train (Colab) — see train/train_qlora.ipynb

```bash
python -m train.qlora_train \
  --data data/v4/gold_v4_r4.jsonl \
  --adapter-name v4r4 \
  --epochs 3 \
  --lora-r 32 --lora-alpha 64 \
  --batch-size 8 --grad-accum 2 --lr 2e-4
```

## 4. Evaluate + judge (does not overwrite v4r3 outputs)

Readability-only on Colab (litmus 12 and full 24):

```bash
python -m eval.base_vs_tuned --adapter train/adapters/v4r4 --eval-key eval_litmus --limit 0 --no-judge --out eval/base_vs_tuned_v4r4_litmus12_readability.json
python -m eval.base_vs_tuned --adapter train/adapters/v4r4 --eval-key eval        --limit 0 --no-judge --out eval/base_vs_tuned_v4r4_all24_readability.json
```

Then add accuracy locally from the saved outputs (no GPU needed):

```bash
python -m eval.judge_saved base_vs_tuned_v4r4_litmus12_readability.json base_vs_tuned_v4r4_all24_readability.json --judge openai-group/gpt-4.1
```

Success target vs v4r3 (litmus 12): accuracy recovers toward >= 11/12 while readability
holds >= ~7-8/12 (overall >= 7/12 beats v4r3's 6/12).
