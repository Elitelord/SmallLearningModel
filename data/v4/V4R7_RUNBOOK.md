# v4r7 clean-union capacity runbook

## Why this run

v4r6 proved that the clean mixed-replay data can preserve tolerant accuracy: it
reached 10/12 accuracy-v2, the best tuned-model score so far. Its conservative
r16/a32, two-epoch, `1e-4` training recipe reached only 5/12 readability, however.

v4r4 remains the readability leader at 9/12. Its successful training recipe used
r32/a64, three epochs, and learning rate `2e-4`. v4r7 transfers that recipe to the
clean, benchmark-preserving data rather than generating toward litmus failures.

## Dataset

`gold_v4_r7.jsonl` is the complete deterministic union of eligible clean records:

- 98 v4r2 accuracy anchors.
- 106 additional v4r4 readability records after cross-source deduplication.
- 281 additional v4r5 clean targets after cross-source deduplication.
- 485 total unique prompts.

Every record passes clean accuracy-v2, the centered v4r3 readability target, the v4
eval gate, and the four-to-six-sentence constraint. Eval, calibration, blind, and
historical failure-targeted prompts remain excluded.

Rebuild and verify with:

```powershell
.venv\Scripts\python.exe -m data.build_v4r6_mix `
  --r2-count 98 --r4-count 0 --r5-count 0 `
  --out data\v4\gold_v4_r7.jsonl

.venv\Scripts\python.exe -m data.audit_v4r3 data\v4\gold_v4_r7.jsonl `
  --accuracy-gate clean-v2 --forbid-targeted-v4r3
```

## Training

Run the first two cells under **Controlled v4r7 clean-union capacity run** in
`train/train_qlora.ipynb`:

- QLoRA r32/alpha64.
- Three epochs.
- Learning rate `2e-4`.
- Batch size 8, gradient accumulation 2.

The first cell verifies the canonical dataset hash, trains, and saves the exact
adapter/data/stats to `MyDrive/SmallLearningModel/v4r7`. The second cell evaluates only
`calibration_v4r5`. The completed calibration selected deterministic temperature `0`:
17/24 readability versus 15.33/24 averages for both `0.3` and `0.7`. Run the single
fixed development-litmus cell at temperature `0`, seed `0`, then copy both litmus JSON
files locally for accuracy-v2 judging. Keep `blind_v4r5` sealed.
