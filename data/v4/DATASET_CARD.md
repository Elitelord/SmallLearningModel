# v4 gold dataset

Training set for the grade-3 science-explainer fine-tune, built under the **v4
readability gate**. Prompt format is the minimal `Explain: {concept}` (see
`data/sft_format.py`) — behavior must live in the weights, not the prompt.

## Files

| File | Records | What |
|---|---|---|
| `gold_v4_final.jsonl` | **228** | **The training file** — synthetic + real, combined. |
| `gold_v4.jsonl` | 227 | Synthetic component (pipeline output; see `.stats.json`). |
| `real_slice_v4.jsonl` | 1 | Real human-written component (CLEAR excerpt, see attribution). |

## Gates (every record passes both)

- **Readability — `readability_pass_v4`** (`litmus/fk_score.py`): whole-passage
  Flesch-Kincaid grade in **3.0–6.0** AND ARI in **3.0–7.0**, per-sentence FK
  std-dev ≤ 1.7, and no ≥10-word sentence over FK 8.0. This targets *genuine*
  grade-3 reading level (recalibrated against the CLEAR corpus — see
  `eval/metric_comparison_real.md`), not the FK~2 baby-talk of the old v3 gate.
- **Accuracy = 2** (`litmus/accuracy.py`): a mechanism-rubric judge (0/1/2) keeps
  only 2s. Judge is a different model family from the teacher.

## Composition

**227 synthetic** — generate → v4-readability-rewrite → accuracy-gate pipeline
(`data/generate.py`):
- teacher `openai-group/gpt-5.4`, judge `openai-group/gpt-4.1` (via TrueFoundry gateway)
- 250 concept/phrasing items attempted → 227 kept (**90.8% yield**), avg 1.56 rewrites
- reproduce: `python -m data.generate --sample 250 --teacher openai-group/gpt-5.4 --judge openai-group/gpt-4.1 --out data/v4/gold_v4.jsonl`

**1 real** — a verbatim (unaltered) excerpt from the CommonLit CLEAR corpus that
independently clears the same v4 + accuracy gates. Only one CLEAR passage
qualified under strict criteria (grade-3 Lexile, elementary-science topic,
self-contained without editing).

## Distribution (228 records)

- unique concepts: 228 / 228 (no duplicates)
- whole-passage FK: 3.05–6.0 (median 4.66, mean 4.62)
- whole-passage ARI: 3.56–6.99
- sentences per example: mostly 5–6 (a few 7–8)

## Attribution

The single real record is an excerpt from **"The Time Traveling River"** (CommonLit
CLEAR Corpus, `clear_id` 2294), used **verbatim** under its **CC BY 4.0** license.
See the `provenance` field in `real_slice_v4.jsonl`. No wording was changed; only a
contiguous sentence range was selected.
