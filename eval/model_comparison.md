# Model comparison — fine-tuned Qwen3-4B vs. litmus baselines

All rows are scored on the **same 12 held-out litmus concepts** with the **same v4
readability gate** (whole-passage FK 3.0–6.0 **AND** ARI 3.0–7.0, dispersion ≤ 1.7,
long-sentence backstop) and the **same accuracy judge** (0/1/2 mechanism rubric,
audience-calibrated). `overall pass = readability_pass_v4 AND accuracy == 2`.

Per-model, per-concept detail lives in `litmus/results_v4.md` (the prompted
baselines) and in the `base_vs_tuned_results.json` runs (the tuned models).

## Headline

| Model | Prompt | readability (v4) | accuracy (=2) | **overall pass** |
|---|---|---|---|---|
| Qwen3-0.6B (base) | full grade-3 prompt | 2/12 | 5/12 | 0/12 |
| Claude (browser) | full grade-3 prompt | 1/12 | 12/12 | 1/12 |
| GPT-4o | full grade-3 prompt | 2/12 | 12/12 | 2/12 |
| Qwen3-4B (**base**) | full grade-3 prompt | 2/12 | 12/12 | 2/12 |
| Gemini (browser) | full grade-3 prompt | 4/12 | 12/12 | 4/12 |
| **Qwen3-4B + v1 tune** (`v4`) | bare `Explain:` | 3/12 | 11/12 | 3/12 |
| **Qwen3-4B + v2 tune** (`v4r2`) | bare `Explain:` | **5/12** | 11/12 | **5/12** |

The fine-tuned 4B, prompted with only `Explain: {concept}`, clears the full grade-3
spec on **more held-out concepts (5/12) than GPT-4o (2), Claude (1), or Gemini (4)
managed with an explicit grade-3 prompt** — and up from the untuned same-model base
(2/12 → 5/12).

## What the numbers say

- **The prompt asymmetry favors the tuned model.** The litmus baselines were given a
  full "explain this for a 3rd grader" prompt; the tuned model gets only `Explain:`.
  Winning with the *weaker* prompt is the project thesis — the behavior lives in the
  **weights**, not the prompt — demonstrated, not asserted.
- **Frontier models fail on readability, not accuracy.** GPT/Claude/Gemini all score
  12/12 accuracy but 1–4/12 readability: they are factually correct but read at roughly
  grade 5–7. That readability gap is exactly what fine-tuning closed and prompting
  could not.
- **The cost is ~1 accuracy point** (tuned 11/12 vs. frontier 12/12) for a large
  readability gain. Net, the v2 tune has the best overall pass-rate here.
- **Capacity mattered.** Qwen3-0.6B (base) manages 5/12 accuracy — it cannot hold the
  mechanism reliably at all, which is why the tune target was upgraded 0.6B → 4B.

## Caveats

- **n = 12.** Directionally strong, not statistically precise. A pass-rate point is
  one concept (±8%).
- **Base rows differ by prompt.** The litmus "Qwen3-4B (base)" row uses the full
  grade-3 prompt (2/12); the base column in the tuned eval uses the bare `Explain:`
  prompt and reads at ~grade 9–10 (0/12). Same base model, different prompt — the
  tuned rows are the meaningful comparison.
- **Readability gate is fixed across all rows** and was recalibrated against the
  independent CLEAR corpus (`eval/metric_comparison_real.md`), so no model — baseline
  or tuned — was scored against a moved goalpost.

## Iteration record

v1 → v2 was a diagnosed fix: v1's held-out failures were readability (overshooting the
band in both directions), accuracy was already fine (92%). v2 trained on data centered
tighter *inside* the eval band (generation target FK 3.5–5.5 / ARI 3.5–6.5), leaving
the eval gate unchanged. Result: readability/overall pass 25% → 42% (3/12 → 5/12) with
accuracy held at 92%.
