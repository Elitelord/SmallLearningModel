# v4r6 accuracy-anchor/readability-replay runbook

## Objective

Train one controlled r16/e2 adapter from 400 strict, clean records:

- 98 v4r2 records as accuracy anchors (all available clean tight records).
- 102 v4r4 records as readability replay.
- 200 v4r5 records as clean readability targets.

Every replay record must pass the centered v4r3 readability target and clean
accuracy-v2. Exact eval, calibration, blind, and historical failure-targeted prompts
remain excluded. Selection is deterministic and never uses concept-level litmus
performance.

## 1. Finish the two source audits locally

The audit inputs are already built. Existing broad-audit judgments are reused by
matching the exact training prompt and output hash.

```powershell
.venv\Scripts\python.exe -m litmus.judge_accuracy_v2 `
  --input data\v4\v4r2_tight_accuracy_audit.json `
  --out data\v4\v4r2_tight_accuracy_audit.scores.json `
  --reuse-from data\v4\v4r2_broad_accuracy_audit.scores.json --resume

.venv\Scripts\python.exe -m litmus.judge_accuracy_v2 `
  --input data\v4\v4r4_tight_accuracy_audit.json `
  --out data\v4\v4r4_tight_accuracy_audit.scores.json `
  --reuse-from data\v4\v4r4_broad_accuracy_audit.scores.json --resume
```

The completed audits produced 98/101 clean tight r2 records and 116/140 clean tight
r4 records. Gemini was used only where the primary judges disagreed.

## 2. Export clean replay and build the mix

```powershell
.venv\Scripts\python.exe -m data.export_clean_replay `
  --input data\v4\v4r2_tight_accuracy_audit.scores.json `
  --out data\v4\v4r2_tight_replay_clean.jsonl

.venv\Scripts\python.exe -m data.export_clean_replay `
  --input data\v4\v4r4_tight_accuracy_audit.scores.json `
  --out data\v4\v4r4_tight_replay_clean.jsonl

.venv\Scripts\python.exe -m data.build_v4r6_mix
```

Do not start Colab unless `gold_v4_r6.jsonl` contains 400 unique records and the
builder completes its strict audit. Commit and push the code, audit outputs, replay
files, `gold_v4_r6.jsonl`, and `gold_v4_r6.stats.json` before running Colab.

## 3. Controlled training and calibration

Run the two cells under **Controlled v4r6 mixed replay** in
`train/train_qlora.ipynb`. The fixed training configuration is:

- QLoRA r16/alpha32
- two epochs
- learning rate `1e-4`
- batch size 8, gradient accumulation 2

The first cell trains and persists the adapter and exact dataset to Drive. The second
cell sweeps only `calibration_v4r5` and persists all calibration outputs. The completed
calibration selected deterministic temperature `0`: 12/24 readability versus 10.33/24
for temperature `0.3` and 10/24 for `0.7`, averaged across their three seeds. Run the
single fixed development-litmus cell at temperature `0`, seed `0`, then copy both
litmus JSON files back locally for accuracy-v2 judging. Keep `blind_v4r5` sealed.
