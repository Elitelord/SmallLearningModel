# Litmus baseline under the v4 gate (FK 3.0-6.0 AND ARI 3.0-7.0, dispersion ≤ 1.7)

Same 12 concepts, same saved outputs, same accuracy judgments as `results_v3.md` (gpt-4o, audience-calibrated). Only the readability gate changed: v3's FK 1.5-3.0 band was shown to target ~grade 1-2 (`eval/metric_comparison_real.md`); v4 uses the recalibrated real-grade-3 band FK 3-6 plus the co-best co-metric ARI 3-7.

`overall_pass = readability_pass_v4 AND accuracy==2`

## Headline (v4)

| Model | readability (v4) | accuracy=2 | overall pass |
|---|---|---|---|
| GPT (gpt-4o) | 2/12 | 12/12 | **2/12** |
| Claude (browser) | 1/12 | 12/12 | **1/12** |
| Gemini (browser) | 4/12 | 12/12 | **4/12** |
| Qwen3-4B (local, instruct-style) | 2/12 | 12/12 | **2/12** |
| Qwen3-0.6B (local, reference) | 2/12 | 5/12 | **0/12** |

## GPT (gpt-4o)  —  `gpt-4o`

| Concept | wpFK | ARI | stdev | FK✓ | ARI✓ | read(v4) | acc | overall |
|---|---|---|---|---|---|---|---|---|
| Why is the sky blue? | 5.04 | 6.09 | 2.97 | ✅ | ✅ | ❌ | 2 | ❌ |
| How do plants make their own food? | 3.66 | 7.55 | 2.71 | ✅ | ❌ | ❌ | 2 | ❌ |
| Why do we have day and night? | 1.82 | 2.65 | 1.26 | ❌ | ❌ | ❌ | 2 | ❌ |
| What makes ice melt? | 2.64 | 3.28 | 1.6 | ❌ | ✅ | ❌ | 2 | ❌ |
| How do magnets work? | 5.04 | 6.48 | 1.88 | ✅ | ✅ | ❌ | 2 | ❌ |
| Why do things fall to the ground? | 4.69 | 4.49 | 1.74 | ✅ | ✅ | ❌ | 2 | ❌ |
| Where does a puddle go when it dries up? | 3.65 | 2.75 | 1.86 | ✅ | ❌ | ❌ | 2 | ❌ |
| Why do we have seasons? | 2.78 | 3.93 | 1.73 | ❌ | ✅ | ❌ | 2 | ❌ |
| How do our lungs help us breathe? | 3.84 | 4.39 | 1.08 | ✅ | ✅ | ✅ | 2 | ✅ |
| What makes a rainbow? | 4.45 | 5.52 | 0.99 | ✅ | ✅ | ✅ | 2 | ✅ |
| Why does the moon look like it changes shape? | 1.86 | 2.79 | 2.04 | ❌ | ❌ | ❌ | 2 | ❌ |
| How do fish breathe underwater? | 3.16 | 2.98 | 2.21 | ✅ | ❌ | ❌ | 2 | ❌ |

## Claude (browser)  —  `browser (manual paste)`

| Concept | wpFK | ARI | stdev | FK✓ | ARI✓ | read(v4) | acc | overall |
|---|---|---|---|---|---|---|---|---|
| Why is the sky blue? | 3.65 | 4.0 | 1.52 | ✅ | ✅ | ✅ | 2 | ✅ |
| How do plants make their own food? | 5.77 | 6.44 | 2.36 | ✅ | ✅ | ❌ | 2 | ❌ |
| Why do we have day and night? | 2.38 | 2.34 | 0.36 | ❌ | ❌ | ❌ | 2 | ❌ |
| What makes ice melt? | 3.64 | 3.69 | 1.85 | ✅ | ✅ | ❌ | 2 | ❌ |
| How do magnets work? | 5.83 | 6.31 | 2.41 | ✅ | ✅ | ❌ | 2 | ❌ |
| Why do things fall to the ground? | 6.24 | 6.14 | 3.01 | ❌ | ✅ | ❌ | 2 | ❌ |
| Where does a puddle go when it dries up? | 5.83 | 4.95 | 1.92 | ✅ | ✅ | ❌ | 2 | ❌ |
| Why do we have seasons? | 6.03 | 7.52 | 1.26 | ❌ | ❌ | ❌ | 2 | ❌ |
| How do our lungs help us breathe? | 3.75 | 4.43 | 2.63 | ✅ | ✅ | ❌ | 2 | ❌ |
| What makes a rainbow? | 6.77 | 7.32 | 1.12 | ❌ | ❌ | ❌ | 2 | ❌ |
| Why does the moon look like it changes shape? | 3.98 | 4.46 | 2.21 | ✅ | ✅ | ❌ | 2 | ❌ |
| How do fish breathe underwater? | 4.96 | 6.52 | 3.0 | ✅ | ✅ | ❌ | 2 | ❌ |

## Gemini (browser)  —  `browser (manual paste)`

| Concept | wpFK | ARI | stdev | FK✓ | ARI✓ | read(v4) | acc | overall |
|---|---|---|---|---|---|---|---|---|
| Why is the sky blue? | 5.86 | 7.15 | 2.29 | ✅ | ❌ | ❌ | 2 | ❌ |
| How do plants make their own food? | 4.43 | 5.54 | 1.61 | ✅ | ✅ | ✅ | 2 | ✅ |
| Why do we have day and night? | 3.2 | 4.71 | 1.24 | ✅ | ✅ | ✅ | 2 | ✅ |
| What makes ice melt? | 5.5 | 5.54 | 1.55 | ✅ | ✅ | ✅ | 2 | ✅ |
| How do magnets work? | 6.99 | 7.92 | 1.12 | ❌ | ❌ | ❌ | 2 | ❌ |
| Why do things fall to the ground? | 7.31 | 8.71 | 2.36 | ❌ | ❌ | ❌ | 2 | ❌ |
| Where does a puddle go when it dries up? | 7.01 | 6.94 | 1.2 | ❌ | ✅ | ❌ | 2 | ❌ |
| Why do we have seasons? | 6.22 | 7.4 | 1.64 | ❌ | ❌ | ❌ | 2 | ❌ |
| How do our lungs help us breathe? | 3.4 | 5.24 | 0.98 | ✅ | ✅ | ✅ | 2 | ✅ |
| What makes a rainbow? | 6.18 | 8.04 | 1.91 | ❌ | ❌ | ❌ | 2 | ❌ |
| Why does the moon look like it changes shape? | 4.91 | 5.36 | 2.37 | ✅ | ✅ | ❌ | 2 | ❌ |
| How do fish breathe underwater? | 5.46 | 5.56 | 1.88 | ✅ | ✅ | ❌ | 2 | ❌ |

## Qwen3-4B (local, instruct-style)  —  `Qwen/Qwen3-4B`

| Concept | wpFK | ARI | stdev | FK✓ | ARI✓ | read(v4) | acc | overall |
|---|---|---|---|---|---|---|---|---|
| Why is the sky blue? | 0.56 | 0.45 | 0.95 | ❌ | ❌ | ❌ | 2 | ❌ |
| How do plants make their own food? | 2.75 | 5.64 | 1.34 | ❌ | ✅ | ❌ | 2 | ❌ |
| Why do we have day and night? | 2.36 | 2.94 | 0.73 | ❌ | ❌ | ❌ | 2 | ❌ |
| What makes ice melt? | 1.52 | 1.97 | 1.05 | ❌ | ❌ | ❌ | 2 | ❌ |
| How do magnets work? | 2.55 | 4.39 | 1.3 | ❌ | ✅ | ❌ | 2 | ❌ |
| Why do things fall to the ground? | 3.88 | 5.34 | 1.68 | ✅ | ✅ | ✅ | 2 | ✅ |
| Where does a puddle go when it dries up? | 2.88 | 1.07 | 1.28 | ❌ | ❌ | ❌ | 2 | ❌ |
| Why do we have seasons? | 2.29 | 3.1 | 1.0 | ❌ | ✅ | ❌ | 2 | ❌ |
| How do our lungs help us breathe? | 3.64 | 2.4 | 2.54 | ✅ | ❌ | ❌ | 2 | ❌ |
| What makes a rainbow? | 3.83 | 4.84 | 1.74 | ✅ | ✅ | ❌ | 2 | ❌ |
| Why does the moon look like it changes shape? | 2.35 | 3.08 | 1.86 | ❌ | ✅ | ❌ | 2 | ❌ |
| How do fish breathe underwater? | 3.57 | 3.22 | 0.9 | ✅ | ✅ | ✅ | 2 | ✅ |

## Qwen3-0.6B (local, reference)  —  `Qwen/Qwen3-0.6B`

| Concept | wpFK | ARI | stdev | FK✓ | ARI✓ | read(v4) | acc | overall |
|---|---|---|---|---|---|---|---|---|
| Why is the sky blue? | 7.25 | 8.38 | 1.98 | ❌ | ❌ | ❌ | 1 | ❌ |
| How do plants make their own food? | 4.96 | 7.14 | 1.96 | ✅ | ❌ | ❌ | 2 | ❌ |
| Why do we have day and night? | 7.35 | 8.43 | 1.25 | ❌ | ❌ | ❌ | 2 | ❌ |
| What makes ice melt? | 8.73 | 9.14 | 3.06 | ❌ | ❌ | ❌ | 0 | ❌ |
| How do magnets work? | 5.86 | 6.94 | 0.41 | ✅ | ✅ | ✅ | 0 | ❌ |
| Why do things fall to the ground? | 5.25 | 6.68 | 1.53 | ✅ | ✅ | ✅ | 1 | ❌ |
| Where does a puddle go when it dries up? | 6.12 | 5.74 | 2.45 | ❌ | ✅ | ❌ | 2 | ❌ |
| Why do we have seasons? | 6.8 | 7.66 | 1.2 | ❌ | ❌ | ❌ | 2 | ❌ |
| How do our lungs help us breathe? | 8.95 | 9.67 | 2.47 | ❌ | ❌ | ❌ | 1 | ❌ |
| What makes a rainbow? | 7.56 | 9.08 | 2.25 | ❌ | ❌ | ❌ | 2 | ❌ |
| Why does the moon look like it changes shape? | 5.35 | 6.43 | 2.49 | ✅ | ✅ | ❌ | 0 | ❌ |
| How do fish breathe underwater? | 7.97 | 8.96 | 1.6 | ❌ | ❌ | ❌ | 0 | ❌ |

## Failure-mode breakdown (why they miss the v4 band)

Accuracy is saturated (all four at 12/12), so **readability is the only
differentiator.** Splitting the readability failures (dispersion cap = 1.7):

| Model | in FK+ARI band | pass v4 (+even) | too simple (<floor) | too hard (>ceiling) | uneven (stdev>1.7) |
|---|---|---|---|---|---|
| GPT | 5/12 | 2/12 | 6 | 1 | 8 |
| Claude | 8/12 | 1/12 | 1 | 3 | 8 |
| Gemini | 6/12 | 4/12 | 0 | 6 | 5 |
| Qwen3-4B | 3/12 | 2/12 | 9 | 0 | 3 |

Reading:
- **Unevenness (dispersion) is still the most common binding constraint** for the
  frontier models (5–8 of 12), even after loosening the cap to 1.7. Passages sit in
  the FK+ARI grade-3 band on average but lurch between a baby-talk sentence and a
  hard one.
- **The models miss in opposite directions.** Qwen3-4B is *too simple* (9/12 below
  the grade-3 floor — terse, sub-grade-3 vocabulary; this is why it looked good under
  the old FK 1.5–3.0 band). Gemini skews *too hard* (6/12 over ceiling). GPT/Claude
  straddle the band but read unevenly.
- **Thesis intact.** Under a band calibrated to *real* grade-3 reading level, no
  prompted model reliably hits grade-3 + accurate + even — best is 4/12 (Gemini). The
  4B fine-tune target sits at 2/12 at baseline. Fine-tuning is still the justified
  move; what changed is the *target* (pull text into FK 3–6 evenly, not down to FK ~2).
