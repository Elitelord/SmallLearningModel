# v4r8 two-epoch midpoint runbook

## Rationale

v4r6 used a conservative r16/a32, two-epoch, `1e-4` recipe and reached 5/12
readability, 10/12 accuracy-v2, and 5/12 overall-v2. v4r7 used the complete 485-record
clean union with r32/a64, three epochs, and `2e-4`; it reached 10/12 readability,
9/12 accuracy-v2, and a new-best 7/12 overall-v2.

v4r8 is a controlled midpoint. It changes only v4r7's training duration from three
epochs to two, keeping the dataset, rank, alpha, learning rate, batching, prompt, and
evaluation policy fixed. The aim is to retain most of v4r7's readability while
recovering the accuracy lost at the stronger training dose.

## Training

Run the first two cells under **Controlled v4r8 two-epoch midpoint** in
`train/train_qlora.ipynb`:

- Dataset: `data/v4/gold_v4_r7.jsonl` (485 clean unique records).
- QLoRA r32/alpha64.
- Two epochs.
- Learning rate `2e-4`.
- Batch size 8, gradient accumulation 2.

The first cell verifies the canonical v4r7 dataset hash and persists the adapter to
`MyDrive/SmallLearningModel/v4r8`. The second cell runs only `calibration_v4r5`.
The completed calibration selected deterministic temperature `0`: 18/24 readability,
versus 12.67/24 for temperature `0.3` and 14.33/24 for `0.7`, averaged across their
three seeds. Run the single fixed development-litmus cell at temperature `0`, seed
`0`, then copy both litmus files locally for accuracy-v2 judging. Keep `blind_v4r5`
sealed.
