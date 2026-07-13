# v4r9 one-and-a-half-epoch ablation runbook

## Goal

v4r8 is the current best tuned model at 9/12 readability, 9/12 accuracy-v2, and
8/12 overall-v2. Its only accuracy failure among readable outputs is seasons; plants
is accurate but narrowly misses the readability-dispersion cap. v4r9 tests whether a
slightly lower training dose can preserve nine or more readable outputs while
recovering one accuracy pass and reaching the 9/12 overall target.

## Controlled change

v4r9 changes only training duration from v4r8's two epochs to **1.5 epochs**:

- Dataset: `data/v4/gold_v4_r7.jsonl` (485 clean unique records).
- Qwen3-4B with QLoRA r32/alpha64.
- Learning rate `2e-4`.
- Batch size 8 and gradient accumulation 2.
- Assistant-only loss and bare `Explain:` inference prompt.
- No litmus-targeted or near-neighbor generation.

## Colab sequence

Run the three code cells under **Controlled v4r9 1.5-epoch ablation** in
`train/train_qlora.ipynb` in order:

1. Train and persist the adapter and frozen dataset to
   `MyDrive/SmallLearningModel/v4r9`.
2. Run the sealed `calibration_v4r5` sweep.
3. Run the original development litmus once with the automatically selected
   calibration temperature and seed 0.

The selection code maximizes mean calibration readability across seeds, then minimizes
mean readability penalty, then prefers the lower temperature. It writes
`eval/v4r9_decode_selection.json` so the choice is auditable. The development litmus
does not influence decoding selection. Keep `blind_v4r5` sealed.

Copy these files from Drive back into local `eval/`:

- `v4r9_decode_calibration.json`
- `v4r9_decode_calibration.top.json`
- `v4r9_decode_selection.json`
- `v4r9_decode_litmus.json`
- `v4r9_decode_litmus.top.json`

## Local accuracy-v2 judging

After copying the litmus files locally, run:

```powershell
.venv\Scripts\python.exe -m litmus.judge_accuracy_v2 `
  --input eval/v4r9_decode_litmus.json `
  --out eval/v4r9_decode_litmus_accuracy_v2.json `
  --resume

.venv\Scripts\python.exe -m eval.summarize_sweep_v2 `
  eval/v4r9_decode_litmus_accuracy_v2.json `
  --out eval/v4r9_decode_litmus_accuracy_v2.summary.json `
  --require-n 12 --require-overall 9
```

Accept v4r9 as the new best only if `overall_pass_v2 >= 9/12`. Do not open or run the
blind holdout merely to choose another training recipe.
