# Readability metrics vs. INDEPENDENT grade-3 ground truth (CLEAR corpus)

**Why this exists.** The first metric study (`metric_comparison.md`) ran on
`fk_eval_drafts_37.jsonl`, whose 37 "grade-3" positives were AI-authored and
*iterated against Flesch-Kincaid*. So FK was structurally advantaged: the eval
asked "which metric best matches the metric we optimised for?" This redo uses
labels **no formula ever saw**.

**Ground truth.** CommonLit **CLEAR corpus** (4,724 excerpts). Each carries a
`Lexile Band` (grade designation) and `BT_easiness` — a Bradley-Terry ease score
from *teacher pairwise judgments*, not a formula. Lexile→grade (MetaMetrics):
band 500 ≈ G2-3, 700 ≈ G3-4, 900 ≈ G4-5, 1100 ≈ G6-8, 1300 ≈ G9-11, 1500 ≈ G11-12.

**Test set** (`data/sample/grade3_real_clear.jsonl`, Info category only, to stay
near the science-explanation domain): 152 real grade-3 positives (band 500/700) +
152 adult negatives (band 1300/1500), balanced. Every metric recomputed with the
repo's textstat 0.7.13 (CLEAR's own precomputed columns discarded).
Reproduce: `eval/build_grade3_real.py` then `eval/run_grade3_real.py`
(raw numbers in `eval/_grade3_real_results.json`).

---

## Finding 1 — The FK 1.5–3.0 gate targets ~grade 1–2, not grade 3

Where genuine grade-3 material actually lands (positives only):

| Metric | mean | median | p10–p90 |
|---|---|---|---|
| **Flesch-Kincaid** | **5.50** | 5.16 | 3.46 – 8.16 |
| ARI | 5.86 | 5.47 | 3.29 – 8.74 |
| Coleman-Liau | 6.85 | 6.63 | 4.48 – 9.52 |
| SMOG | 8.55 | 8.15 | 6.63 – 10.89 |
| Dale-Chall | 8.17 | 8.07 | 6.69 – 9.85 |

**Only 5.9% of real grade-3 informational prose passes FK ≤ 3.0.** The project's
operative band (whole-passage FK 1.5–3.0) sits ~2.5 FK grades *below* where real
grade-3 text lives.

Per-Lexile-band gradient (mean, Info excerpts) — every metric climbs monotonically
with the independent grade label, and the FK=3 line falls around **band 300 (~G1-2)**:

| Lexile band | ~grade | n | FK | ARI | Coleman-Liau | SMOG | Dale-Chall |
|---|---|---|---|---|---|---|---|
| 300 | ~G1-2 | 1 | 2.25 | 3.03 | 4.76 | 7.17 | 7.18 |
| **500** | **~G2-3** | 31 | **4.38** | 4.75 | 6.26 | 7.47 | 7.83 |
| **700** | **~G3-4** | 89 | **6.38** | 6.66 | 7.29 | 9.27 | 8.36 |
| 900 | ~G4-5 | 299 | 8.19 | 8.74 | 8.77 | 10.75 | 9.01 |
| 1100 | ~G6-8 | 712 | 10.16 | 11.04 | 9.76 | 12.12 | 9.50 |
| 1300 | ~G9-11 | 915 | 12.72 | 14.20 | 10.71 | 13.81 | 10.05 |
| 1500 | ~G11-12 | 166 | 15.36 | 17.03 | 12.01 | 15.91 | 10.91 |

### It is vocabulary, not sentence length (confound ruled out)

The project's own texts scored in the same frame:

| Set | n | FK | ARI | words/sent | sents | words |
|---|---|---|---|---|---|---|
| **v1 gold (project target)** | 90 | **2.35** | 3.12 | 11.0 | 5.0 | 54.9 |
| exemplars (hand-authored) | 4 | 2.45 | 3.63 | 11.1 | 4.8 | 52.8 |
| old `authored_grade3` (FK-tuned) | 37 | 2.39 | 2.88 | 10.2 | 5.0 | 51.1 |
| old `frontier_litmus` (neg) | 24 | 4.55 | 5.46 | 12.6 | 7.1 | 88.0 |
| **CLEAR real grade-3** | 152 | **5.50** | 5.86 | 11.8 | 16.0 | 176.3 |

v1 gold and CLEAR grade-3 have **near-identical words-per-sentence (11.0 vs 11.8)**
yet FK 2.35 vs 5.50. FK's sentence-length term is therefore *not* the driver — the
gap is the **syllables-per-word** term. Real grade-3 informational prose uses richer
vocabulary (photosynthesis, temperature, ancient…) than the project's deliberately
plain explanations. So v1 gold is genuinely *simpler* than real grade-3 text —
roughly grade 1–2 vocabulary at grade-3 sentence length.

> **Note on the old eval:** its "negatives" (frontier_litmus, FK 4.55) were *easier*
> than real grade-3 material (FK 5.50), and its positives (FK 2.39) sat where v1 gold
> sits. The old separation task was both circular *and* mislabeled — adult-simplified
> text was the "hard" class while genuinely harder grade-3 prose wasn't present at all.

---

## Finding 2 — On honest labels, FK and ARI are co-best (this *confirms* the gate, non-circularly)

Single-signal separation, grade-3 vs adult, independent labels:

| Metric | AUC | best thr (≤) | errors/304 | FP | FN | F1 |
|---|---|---|---|---|---|---|
| **Flesch-Kincaid** | **0.990** | 9.49 | 11 | 7 | 4 | 0.964 |
| **ARI** | **0.988** | 9.92 | 10 | 2 | 8 | 0.966 |
| SMOG | 0.975 | 11.45 | 22 | 17 | 5 | 0.930 |
| Coleman-Liau | 0.923 | 8.71 | 46 | 19 | 27 | 0.845 |
| Dale-Chall | 0.876 | 9.15 | 63 | 32 | 31 | 0.793 |

FK and ARI are statistically tied at the top (AUC ≈ 0.99). SMOG is a step behind;
**Coleman-Liau and Dale-Chall are clearly worse** — Dale-Chall again the weakest
(its familiar-word list penalises ordinary informational vocabulary). This is the
same ordering as the circular eval, now on labels no metric was tuned on — so the
original choice of FK as the primary ruler is **vindicated, not overturned.**

### Does any formula literally score grade-3 text *as* grade 3?

"Grade-3 window" = grade output in [2, 4]:

| Metric | % of real grade-3 inside [2,4] | % of adult inside [2,4] |
|---|---|---|
| Flesch-Kincaid | 21.1% | 0.0% |
| ARI | 17.1% | 0.0% |
| Coleman-Liau | 4.6% | 0.0% |
| SMOG | 0.0% | 0.0% |

**No.** Every formula systematically *over-grades* simple informational prose by
~2–3 levels, so none places real grade-3 text at "3." They are good **relative**
rankers (Finding 2) but poorly **calibrated** in absolute grade units on this
material.

### Combinations don't beat FK / FK+ARI

| Rule | errors/304 | F1 | note |
|---|---|---|---|
| FK-alone (FK ≤ 9.49) | 11 | 0.964 | baseline |
| FK ≤ 9.49 AND ARI ≤ 9.91 | 10 | 0.966 | training-fit |
| FK AND {Coleman-Liau / SMOG / Dale-Chall} | 11 | 0.964 | no gain |

Leave-one-out CV logistic (honest): FK-alone 15 errors → **FK+ARI 13 (best)** →
4-feature 18 → 5-feature 15. Adding metrics past FK+ARI overfits; Dale-Chall drags.
Same null result as before: **FK-alone is the right simple gate; ARI is its only
worthwhile companion; nothing else earns its complexity.**

---

## Recommendations

1. **Keep Flesch-Kincaid as the primary readability ruler** — now justified on
   independent labels, not on FK-tuned data. If one companion metric is ever added,
   make it **ARI** (co-best, char-based so it fails differently); skip Coleman-Liau,
   SMOG, and especially Dale-Chall for grade-3 detection.

2. **Decide the target band deliberately — it is a product choice, not a bug.**
   - Real grade-3 *reading material* sits at **FK ≈ 4–8** (IQR ~3.5–8).
   - The current gate (FK 1.5–3.0) yields text at **~grade 1–2 vocabulary**.
   - The Behavior Spec says "a 7-year-old can read every sentence alone," which
     *argues for* the simpler end — but if the goal is to match genuine grade-3
     level, the band should move up to roughly **FK 3.0–6.0** (keeps sentences short,
     allows real grade-3 vocabulary). This is the open decision for the next dataset.

3. **Metric ≠ gate calibration.** The formulas rank difficulty well but read ~2–3
   grades high on plain prose; don't read their absolute numbers as true grade level.
