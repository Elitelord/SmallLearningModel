# Model comparison — fine-tuned Qwen3-4B vs. litmus baselines

All rows are scored on the **same 12 held-out litmus concepts** with the **same v4
readability gate** (whole-passage FK 3.0–6.0 **AND** ARI 3.0–7.0, dispersion ≤ 1.7,
long-sentence backstop) and the **same accuracy judge** (0/1/2 mechanism rubric,
audience-calibrated). `overall pass = readability_pass_v4 AND accuracy == 2`.

Per-model, per-concept detail lives in `litmus/results_v4.md` (the prompted
baselines) and in the `base_vs_tuned_*` results JSONs (the tuned models); the v4r3
row is scored in `base_vs_tuned_v4r3_litmus12_judged.json`.

## Headline (12 litmus concepts)

| Model | Prompt | readability (v4) | accuracy (=2) | **overall pass** |
|---|---|---|---|---|
| Qwen3-0.6B (base) | full grade-3 prompt | 2/12 | 5/12 | 0/12 |
| Claude (browser) | full grade-3 prompt | 1/12 | 12/12 | 1/12 |
| GPT-4o | full grade-3 prompt | 2/12 | 12/12 | 2/12 |
| Qwen3-4B (**base**) | full grade-3 prompt | 2/12 | 12/12 | 2/12 |
| Gemini (browser) | full grade-3 prompt | 4/12 | 12/12 | 4/12 |
| Qwen3-4B + v1 tune (`v4`) | bare `Explain:` | 3/12 | 11/12 | 3/12 |
| Qwen3-4B + v2 tune (`v4r2`) | bare `Explain:` | 5/12 | 11/12 | 5/12 |
| **Qwen3-4B + v3 tune** (`v4r3`) | bare `Explain:` | **8/12** | 9/12 | **6/12** |

The latest fine-tuned 4B (**v3 / `v4r3`**), prompted with only `Explain: {concept}`,
clears the full grade-3 spec on **6/12** held-out litmus concepts — more than GPT-4o
(2), Claude (1), or Gemini (4) reached *with* an explicit grade-3 prompt, and the best
overall pass-rate of any row here. Readability rose sharply (5/12 → 8/12); the trade was
two accuracy points (11/12 → 9/12).

### Full held-out set (24 concepts)

On the larger 24-concept held-out split (`base_vs_tuned_v4r3_all24_judged.json`), v4r3
tuned scores **readability 15/24, accuracy 21/24, overall 14/24 (58%)** — vs the bare-prompt
base at 0/24. Same direction as the litmus-12 slice, on 2× the concepts.

## What the numbers say

- **The prompt asymmetry favors the tuned model.** The litmus baselines were given a
  full "explain this for a 3rd grader" prompt; the tuned model gets only `Explain:`.
  Winning with the *weaker* prompt is the project thesis — the behavior lives in the
  **weights**, not the prompt — demonstrated, not asserted.
- **Frontier models fail on readability, not accuracy.** GPT/Claude/Gemini all score
  12/12 accuracy but 1–4/12 readability: they are factually correct but read at roughly
  grade 5–7. That readability gap is exactly what fine-tuning closed and prompting
  could not.
- **Readability and accuracy trade off.** v2 held accuracy at 11/12 with 5/12
  readability; v3 (`v4r3`) pushed readability to 8/12 but accuracy slipped to 9/12. The
  tighter generation band (FK 3.3–5.0, dispersion ≤ 1.1) drove harder simplification —
  and on one concept ("how do fish breathe underwater") that simplification became an
  outright error (accuracy 0). Net overall pass still improved (5/12 → 6/12), but the
  accuracy regression is the thing to watch next.
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

v2 → v3 (`v4r3`) tightened the generation target again (FK 3.3–5.0, ARI 3.8–6.2,
dispersion ≤ 1.1) and scaled the set to 457 examples. Readability jumped 5/12 → 8/12 and
overall 5/12 → 6/12, but accuracy dipped 11/12 → 9/12 (75%), including one
factual-error oversimplification. The tighter band buys readability at a small but real
accuracy cost. Next iteration should protect the mechanism while keeping the readability
gains — e.g. a stronger accuracy-gate emphasis, or a rewrite instruction that forbids
dropping the causal "why" when simplifying. Remaining readability misses are almost all
dispersion (one spiky sentence), not whole-passage level.
