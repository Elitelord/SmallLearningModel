# v4r5 benchmark-preserving runbook

## Evaluation policy

The 12 litmus prompts are part of the 24 explicit eval prompts. Do not generate,
rewrite, filter, or weight training examples from failures on either set. Do not use
either set to choose temperature, LoRA settings, epochs, checkpoints, or teachers.

Earlier v4r3/v4r4 work did include eight failure-informed near-neighbor prompts. That
means the existing litmus score is useful as a regression metric, but it is no longer a
fully blind generalization claim. v4r5 stops adding that leakage; a new untouched
holdout is still required for a clean final claim.

The fixed `calibration_v4r5` prompts are broad and separate from all 24 eval prompts.
Use only this calibration set for decoding and training-setting decisions. The 24
`blind_v4r5` prompts are frozen now and must not be run until every v4r5 decision is
locked. Eval, calibration, and blind prompts are all reserved from data generation.

## 1. Decode calibration

If the v4r4 adapter still exists, load it once and sweep the calibration prompts:

```bash
python -m eval.tuned_sweep \
  --adapter train/adapters/v4r4 \
  --eval-key calibration_v4r5 \
  --temperatures 0,0.3,0.7 --seeds 0,1,2 \
  --top-settings 7 \
  --out eval/v4r4_decode_calibration.json
```

Choose temperature by average readability across every seed for that temperature.
Never choose the luckiest seed. Keep seed `0` predeclared for the frozen run.

Run the 12 litmus prompts once with that fixed setting:

```bash
python -m eval.tuned_sweep \
  --adapter train/adapters/v4r4 \
  --eval-key eval_litmus --final-eval \
  --temperatures CHOSEN_TEMPERATURE --seeds 0 \
  --top-settings 1 \
  --out eval/v4r4_decode_litmus.json
```

Judge every saved output; do not retain only favorable generations:

```powershell
.venv\Scripts\python.exe -m litmus.judge_accuracy_v2 `
  --input eval\v4r4_decode_litmus.top.json `
  --out eval\v4r4_decode_litmus_accuracy_v2.json --resume
.venv\Scripts\python.exe -m eval.summarize_sweep_v2 `
  eval\v4r4_decode_litmus_accuracy_v2.json
```

## 2. Broad data audit

Do not audit only known failure families. Draw a deterministic broad slice from v4r4;
the builder excludes the eight historical failure-targeted prompts:

```powershell
.venv\Scripts\python.exe -m data.build_v4r5_audit --sample-size 40
.venv\Scripts\python.exe -m litmus.judge_accuracy_v2 `
  --input data\v4\v4r4_broad_accuracy_audit.json `
  --out data\v4\v4r4_broad_accuracy_audit.scores.json --resume
```

Use the broad clean-pass and misconception rates to decide whether old records are safe
for replay. Do not use concept-level litmus errors to select replay records.

The completed audit found 30/40 clean, 39/40 tolerant, and one major mechanism error.
Do not replay v4r4 wholesale. Export only the audited consensus-clean records:

```powershell
.venv\Scripts\python.exe -m data.export_clean_replay
```

## 3. Controlled training

If decoding remains weak, change one training variable at a time. A same-data v4r4
retrain is diagnostic only because the old dataset contains failure-informed examples;
do not present it as a new blind result.

For the clean v4r5 loop, generate from the broad train pool only. Target 350-450 unique
records rather than scaling past 605. The generator now omits targeted prompts by
default and excludes all eval and calibration prompts exactly.

- Require the centered readability gate on every target.
- Use mixed Claude/Gemini teachers and opposite-family rewrites.
- Use GPT-5.4 for the online generation gate.
- Run the multi-judge accuracy-v2 rubric on a broad random audit before training.
- Rejudge any record after a rewrite; never reuse a stale accuracy score.
- Do not oversample magnets, seasons, moon phases, gravity, fish, or any other observed
  litmus failure family.

Start with a small paid pilot. This uses both primary accuracy-v2 judges for every
readable draft, Gemini only on disagreement, and requires clean F3/M2 consensus:

```powershell
.venv\Scripts\python.exe -m data.generate `
  --teachers claude-group/claude-sonnet-4-6,gemini-group/gemini-3.1-pro `
  --accuracy-gate clean-v2 --max-accuracy-repairs 2 `
  --seed data\v4\v4r4_broad_replay_clean.jsonl `
  --sample 40 --target-kept 50 --concurrency 4 `
  --out data\v4\gold_v4_r5.jsonl
```

Review the pilot yield and repair count. If healthy, resume over the broad pool:

```powershell
.venv\Scripts\python.exe -m data.generate `
  --teachers claude-group/claude-sonnet-4-6,gemini-group/gemini-3.1-pro `
  --accuracy-gate clean-v2 --max-accuracy-repairs 2 `
  --seed data\v4\v4r4_broad_replay_clean.jsonl `
  --target-kept 400 --concurrency 4 --resume `
  --out data\v4\gold_v4_r5.jsonl
```

Train the first clean adapter with one conservative configuration:

```bash
python -m train.qlora_train \
  --data data/v4/gold_v4_r5.jsonl \
  --adapter-name v4r5 \
  --epochs 2 --lora-r 16 --lora-alpha 32 \
  --batch-size 8 --grad-accum 2 --lr 1e-4
```

Use `calibration_v4r5` for checkpoint and temperature decisions. Then run one frozen
litmus evaluation for regression tracking. Finally, run `blind_v4r5` exactly once with
`--final-eval`, the same fixed decoding settings, and no later model changes. The
predeclared blind target equivalent to 9/12 is overall-v2 at least 18/24.
