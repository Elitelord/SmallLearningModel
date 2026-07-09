# eval/ — base-vs-tuned harness (Part D)

Reuses the litmus eval harness (no rebuild): the FK scorer
(`litmus.fk_score.score_text`) and the accuracy judge
(`litmus.accuracy`). `overall_pass = readability_pass AND accuracy == 2`.

## `base_vs_tuned.py`

Runs **base** Qwen3-0.6B and the **tuned** model (base + LoRA adapter) on the
held-out eval concepts (`data/concepts.json` → `eval`) using the **same minimal
prompt** as training (`data.sft_format`), scores both, and prints a base-vs-tuned
table with per-concept rows and a delta summary.

```
# smoke: 5 held-out concepts, API judge on
.venv\Scripts\python -m eval.base_vs_tuned --adapter train/adapters/smoke --limit 5
# FK only, no API judge (pure offline)
.venv\Scripts\python -m eval.base_vs_tuned --adapter train/adapters/smoke --limit 5 --no-judge
```

Writes `base_vs_tuned_results.json` (per-concept scores + raw model outputs).

## Why the same minimal prompt?

The whole point (Part B) is that the grade-3 behavior lives in the **weights**,
not the prompt. Base and tuned get the identical bare `Explain: {concept}`
instruction; any gap between them is attributable to the fine-tune, not clever
prompting. This is the opposite of the litmus prompt, which was deliberately
verbose to test "can prompting alone do it?".

## Day-2 smoke result (junk data, 10 CPU steps — plumbing only)

The loop completed and printed the table. Even on throwaway data the tuned
adapter already moved readability the right way (avg max_fk 12.4 → 7.8,
%-in-band +0.18), confirming the fine-tune is genuinely wired to the eval. No
`overall_pass` yet — expected; real numbers come from the Day-3 v1 dataset and a
full GPU training run.
