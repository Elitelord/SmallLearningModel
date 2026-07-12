# Model comparison — fine-tuned Qwen3-4B vs. litmus baselines

<!-- accuracy-v2:start -->
## Accuracy-v2 Multi-Judge Results

Rubric `accuracy_v2` uses factuality 0–3 and mechanism 0–2. A clean pass is 3/2; the benchmark accuracy pass allows a minor localized error (factuality ≥2) but still requires mechanism 2. Overall pass also requires the unchanged v4 readability gate.

Primary judges: `openai-group/gpt-5.4` and `claude-group/claude-opus-4-7`. `gemini-group/gemini-3.1-pro` is called only when either primary axis differs; consensus is the per-axis median.

### Headline

| Model | Prompt | Readability | Clean 3/2 | Accuracy-v2 | Overall-v2 | Mean F/M | Gemini |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-4o | full grade-3 prompt | 2/12 | 12/12 | 12/12 | **2/12** | 3.0/2.0 | 4/12 |
| Claude (browser) | full grade-3 prompt | 1/12 | 11/12 | 12/12 | **1/12** | 2.917/2.0 | 2/12 |
| Gemini (browser) | full grade-3 prompt | 4/12 | 8/12 | 12/12 | **4/12** | 2.667/2.0 | 2/12 |
| Qwen3-4B (base) | full grade-3 prompt | 2/12 | 4/12 | 9/12 | **2/12** | 2.333/1.75 | 9/12 |
| Qwen3-0.6B (reference) | full grade-3 prompt | 2/12 | 1/12 | 1/12 | **0/12** | 1.333/0.833 | 7/12 |
| Qwen3-4B + v2 tune (v4r2) | bare Explain: | 5/12 | 7/12 | 9/12 | **4/12** | 2.333/1.667 | 3/12 |
| Qwen3-4B + v3 tune (v4r3) | bare Explain: | 8/12 | 4/12 | 7/12 | **5/12** | 2.0/1.5 | 7/12 |
| Qwen3-4B + v4 tune (v4r4) | bare Explain: | 9/12 | 3/12 | 7/12 | **5/12** | 1.667/1.333 | 6/12 |
| Qwen3-4B + v5 tune (v4r5) | bare Explain: | 7/12 | 3/12 | 8/12 | **3/12** | 1.917/1.583 | 7/12 |
| Qwen3-4B + v6 tune (v4r6) | bare Explain: | 5/12 | 4/12 | **10/12** | **5/12** | 2.167/1.833 | 5/12 |

### Tuned Iteration Comparison

| Iteration | Readability | Clean 3/2 | Accuracy-v2 | Overall-v2 | Mean F/M |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B + v2 tune (v4r2) | 5/12 | 7/12 | 9/12 | **4/12** | 2.333/1.667 |
| Qwen3-4B + v3 tune (v4r3) | 8/12 | 4/12 | 7/12 | **5/12** | 2.0/1.5 |
| Qwen3-4B + v4 tune (v4r4) | 9/12 | 3/12 | 7/12 | **5/12** | 1.667/1.333 |
| Qwen3-4B + v5 tune (v4r5) | 7/12 | 3/12 | 8/12 | **3/12** | 1.917/1.583 |
| Qwen3-4B + v6 tune (v4r6) | 5/12 | 4/12 | **10/12** | **5/12** | 2.167/1.833 |

### v4r5 Regression

v4r5 improves tolerant accuracy by one pass versus v4r4, but loses two readability
passes and two overall passes. Its three overall passes are day/night, lungs, and fish.
The clean multi-judge data gate improved training-target quality, but the conservative
r16/two-epoch run did not reliably learn even readability or preserve several core
mechanisms. The new `blind_v4r5` holdout remains unrun.

### v4r6 Mixed-Replay Result

v4r6 combines 98 clean tight v4r2 accuracy anchors, 102 clean tight v4r4
readability records, and 200 clean v4r5 targets. It raises tolerant accuracy to
**10/12**, the strongest tuned-model result, with only rainbow and moon phases failing.
Readability falls to **5/12**, however, so overall-v2 reaches only **5/12**. The five
overall passes are sky, ice, gravity, puddles, and lungs. The accuracy-anchor strategy
worked, but the conservative r16/two-epoch recipe did not retain r4's readability.
Calibration readability (12/24 at temperature 0) also overstated development-litmus
readability, so calibration is useful for decoding selection but not a sufficient
progression proxy by itself. Raw judgments are in
`eval/v4r6_decode_litmus_accuracy_v2.json`; `blind_v4r5` remains sealed.

### Judge Agreement

The aggregate agreement figures below cover the original 96-output matrix; later
v4r5 and v4r6 iteration rows were judged separately with the same rubric and models.

- Exact two-axis agreement: 56/96 (58.3%).
- Accuracy-pass agreement: 76/96 (79.2%).
- Linear weighted kappa: factuality 0.651, mechanism 0.526.
- Gemini tiebreakers: 40/96.
- Judge-family relationships are recorded per output in the raw JSON; cross-family agreement should be preferred when interpreting tested GPT, Claude, or Gemini rows.
<!-- accuracy-v2:end -->

---

## Historical Accuracy-v1 Results

The following tables are preserved from the original 0/1/2 accuracy rubric.

<!-- accuracy-v1-historical:start -->
All rows use the **same 12 held-out litmus concepts**, the **same v4 readability gate**
(whole-passage FK 3.0–6.0 **AND** ARI 3.0–7.0, dispersion ≤ 1.7, long-sentence
backstop), and the same audience-calibrated 0/1/2 mechanism rubric. The historical
cross-model table retains its original judgments; a stricter GPT-5.4 same-pass audit of
v4r3 and v4r4 appears below. `overall pass = readability_pass_v4 AND accuracy == 2`.

Per-model, per-concept detail lives in `litmus/results_v4.md` (the prompted
baselines) and in the `base_vs_tuned_*` results JSONs (the tuned models). The latest
rows use `base_vs_tuned_v4r3_litmus12_judged.json` and
`base_vs_tuned_v4r4_litmus12_judged.json`.

## Headline (12 litmus concepts, historical judge set)

| Model | Prompt | readability (v4) | accuracy (=2) | **overall pass** |
|---|---|---|---|---|
| Qwen3-0.6B (base) | full grade-3 prompt | 2/12 | 5/12 | 0/12 |
| Claude (browser) | full grade-3 prompt | 1/12 | 12/12 | 1/12 |
| GPT-4o | full grade-3 prompt | 2/12 | 12/12 | 2/12 |
| Qwen3-4B (**base**) | full grade-3 prompt | 2/12 | 12/12 | 2/12 |
| Gemini (browser) | full grade-3 prompt | 4/12 | 12/12 | 4/12 |
| Qwen3-4B + v1 tune (`v4`) | bare `Explain:` | 3/12 | 11/12 | 3/12 |
| Qwen3-4B + v2 tune (`v4r2`) | bare `Explain:` | 5/12 | 11/12 | 5/12 |
| Qwen3-4B + v3 tune (`v4r3`) | bare `Explain:` | 8/12 | 9/12 | **6/12** |
| **Qwen3-4B + v4 tune** (`v4r4`) | bare `Explain:` | **9/12** | 9/12 | **6/12** |

The latest fine-tuned 4B (**v4 / `v4r4`**), prompted with only `Explain: {concept}`,
clears the full grade-3 spec on **6/12** held-out litmus concepts — tied with v4r3 for
the best overall result here and above GPT-4o (2), Claude (1), or Gemini (4), which all
received an explicit grade-3 prompt. It sets the best readability result at **9/12**,
but the mechanism-preserving data loop did not improve the 9/12 accuracy total.

## v4r4 versus v4r3

### Judge comparison

The first two rows are the separately generated/judged litmus artifacts used in the
historical headline. The same-file rows take the first 12 texts from each 24-item run,
which removes duplicate-run judge variance and gives the fairest direct comparison.

| Judge and sample | Run | Readability | Accuracy (=2) | Overall pass |
|---|---|---:|---:|---:|
| GPT-4.1, separate litmus file | v4r3 | 8/12 | 9/12 | 6/12 |
| GPT-4.1, separate litmus file | **v4r4** | **9/12 (+1)** | **9/12 (=)** | **6/12 (=)** |
| GPT-4.1, first 12 of all-24 | v4r3 | 8/12 | **11/12** | **7/12** |
| GPT-4.1, first 12 of all-24 | **v4r4** | **9/12 (+1)** | 9/12 (-2) | 6/12 (-1) |
| **GPT-5.4, first 12 of all-24** | v4r3 | 8/12 | **8/12** | **5/12** |
| **GPT-5.4, first 12 of all-24** | **v4r4** | **9/12 (+1)** | 6/12 (-2) | 4/12 (-1) |
| GPT-4.1, full 24 | v4r3 | 15/24 | **21/24** | 14/24 |
| GPT-4.1, full 24 | **v4r4** | **21/24 (+6)** | 19/24 (-2) | **16/24 (+2)** |
| **GPT-5.4, full 24** | v4r3 | 15/24 | **16/24** | 11/24 |
| **GPT-5.4, full 24** | **v4r4** | **21/24 (+6)** | 14/24 (-2) | **12/24 (+1)** |

GPT-5.4 is substantially stricter than GPT-4.1, lowering full-set accuracy by five
passes for both runs. The relative result is unusually stable: **both judges find the
same two-pass v4r4 accuracy regression on all 24**, while v4r4's six readability gains
still produce a small net overall improvement.

### GPT-5.4 litmus transitions

`R/A/O` means readability pass / accuracy score / overall pass. These are the first 12
rows from each GPT-5.4 all-24 artifact, judged in one pass per iteration.

| Concept | v4r3 R/A/O | v4r4 R/A/O | Change |
|---|---:|---:|---|
| Sky blue | ✓ / 0 / ✗ | ✓ / 2 / ✓ | Accuracy + overall gain |
| Plant food | ✓ / 2 / ✓ | ✓ / 2 / ✓ | Unchanged pass |
| Day and night | ✓ / 2 / ✓ | ✗ / 2 / ✗ | Readability regression |
| Ice melting | ✓ / 2 / ✓ | ✗ / 0 / ✗ | Readability + factual regression |
| Magnets | ✗ / 2 / ✗ | ✓ / 0 / ✗ | Readability gain, factual regression |
| Falling objects | ✗ / 2 / ✗ | ✓ / 0 / ✗ | Readability gain, factual regression |
| Puddle drying | ✓ / 0 / ✗ | ✓ / 2 / ✓ | Accuracy + overall gain |
| Seasons | ✓ / 2 / ✓ | ✓ / 0 / ✗ | Accuracy + overall regression |
| Lungs | ✗ / 2 / ✗ | ✓ / 2 / ✓ | Readability + overall gain |
| Rainbow | ✓ / 2 / ✓ | ✓ / 0 / ✗ | Accuracy + overall regression |
| Moon phases | ✗ / 1 / ✗ | ✓ / 0 / ✗ | Readability gain, factual regression |
| Fish breathing | ✓ / 0 / ✗ | ✗ / 2 / ✗ | Accuracy gain, readability regression |

Under GPT-5.4, sky, puddles, and lungs enter the overall-pass set; day/night, ice,
seasons, and rainbows leave. The strict judge flags six v4r4 litmus explanations as
mechanistically wrong: ice, magnets, falling objects, seasons, rainbows, and moon phases.

### Full held-out set (24 concepts)

On the larger split, GPT-4.1 gives v4r4 **21/24 readability, 19/24 accuracy, and 16/24
overall (67%)**. GPT-5.4 gives the same texts **21/24 readability, 14/24 accuracy, and
12/24 overall (50%)**. Relative to v4r3 under the same judge, that is +6 readability,
-2 accuracy, and +1 overall. The stronger result files are
`base_vs_tuned_v4r3_all24_judged_gpt54.json` and
`base_vs_tuned_v4r4_all24_judged_gpt54.json`.

## What the numbers say

- **The prompt asymmetry favors the tuned model.** The litmus baselines were given a
  full "explain this for a 3rd grader" prompt; the tuned model gets only `Explain:`.
  Winning with the *weaker* prompt is the project thesis — the behavior lives in the
  **weights**, not the prompt — demonstrated, not asserted.
- **Frontier models historically failed on readability, not accuracy.** Under the
  original judge, GPT/Claude/Gemini scored 12/12 accuracy but only 1–4/12 readability.
  They have not yet been rescored by GPT-5.4, so their accuracy totals are not directly
  comparable to the strict v4r3/v4r4 audit.
- **Readability and accuracy still trade off.** Both GPT-4.1 and GPT-5.4 find v4r4 down
  two full-accuracy passes on the 24-item set. GPT-5.4 scores v4r4 at 14/24 accuracy,
  catching plausible-sounding but false causal details that GPT-4.1 accepted.
- **The data gate was only as strong as its judge.** GPT-4.1 marked every retained
  v4r4 training record accurate, but the GPT-5.4 evaluation shows that this did not
  produce reliable mechanism retention at inference. A stronger or consensus data audit
  is a higher-priority next step than generating still more examples.
- **The v4r4 experiment is confounded.** It changed generation/repair behavior, dataset
  size (457 → 605), and LoRA capacity (r16/a32 → r32/a64) together. The result proves the
  combined recipe strengthens readability, but cannot identify which change caused the
  accuracy regression.
- **Capacity mattered.** Qwen3-0.6B (base) manages 5/12 accuracy — it cannot hold the
  mechanism reliably at all, which is why the tune target was upgraded 0.6B → 4B.

## Caveats

- **n = 12.** Directionally strong, not statistically precise. A pass-rate point is
  one concept (±8%).
- **The accuracy judge has visible variance.** The identical v4r3 sky and moon texts
  received scores 1 in the litmus file but 2 in the all-24 file. Tables report each
  saved file as judged; small accuracy deltas should not be treated as exact without
  repeated or consensus judging. The GPT-5.4 comparison avoids this particular problem
  by deriving both litmus subsets from one all-24 pass per iteration.
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
factual-error oversimplification. The tighter band bought readability at a small but real
accuracy cost.

v3 → v4 (`v4r4`) added mechanism-repair rewrites, regenerated 605 jointly gated examples,
and doubled LoRA rank from 16 to 32. Readability improved again (8/12 → 9/12; 15/24 →
21/24). On the fair same-file comparison, accuracy regressed by two under both judges:
GPT-4.1 scores 21/24 → 19/24 and GPT-5.4 scores 16/24 → 14/24. GPT-5.4 overall falls
5/12 → 4/12 on the litmus subset but rises 11/24 → 12/24 on the full set. The next
experiment should change one variable at a time and directly constrain common false
causal patterns, rather than scaling the same recipe further.
<!-- accuracy-v1-historical:end -->
