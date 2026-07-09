# Readability Formula Comparison — Does anything beat Flesch-Kincaid at spotting grade-3 text?

**Question:** Among deterministic readability formulas, which best separates genuine grade-3 text
from adult-level text, and does adding Dale-Chall as a second gate improve on Flesch-Kincaid (FK) alone?

**Scope:** Bounded evaluation of *existing* formulas only. No novel metric.

## Evaluation set
`data/sample/fk_eval_drafts_37.jsonl` — 61 passages, source-based (formula-independent) labels:
- 37 positives = `authored_grade3` (`label_good = true`, verified grade-3)
- 24 negatives = `frontier_litmus` (`label_good = false`, adult-level model output)

All five formulas score *lower = easier*, so the rule tested is **predict grade-3 if `signal <= threshold`**.
"Best threshold" = the cut that minimises total misclassifications (ties broken by F1). AUC is the
threshold-independent Mann-Whitney separation (1.0 = perfect, 0.5 = chance).

## 1. Single-signal separation

| Formula | AUC | Best thr (`<=`) | Errors (of 61) | FP | FN | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| **ARI** (char-based) | **0.892** | 4.45 | **7** | 7 | 0 | 0.885 | 0.841 | 1.000 | **0.914** |
| Coleman-Liau (char-based) | 0.872 | 5.27 | 8 | 5 | 3 | 0.869 | 0.872 | 0.919 | 0.895 |
| SMOG | 0.865 | 6.81 | 8 | 7 | 1 | 0.869 | 0.837 | 0.973 | 0.900 |
| **Flesch-Kincaid** (baseline) | 0.889 | 3.75 | 9 | 9 | 0 | 0.852 | 0.804 | 1.000 | 0.892 |
| **Dale-Chall** (top hypothesis) | **0.660** | 7.29 | **19** | 12 | 7 | 0.689 | 0.714 | 0.811 | 0.759 |

**Findings**
- The four grade-level formulas (ARI, Coleman-Liau, SMOG, FK) cluster tightly: AUC 0.865-0.892,
  7-9 errors. Differences are within ~2 misclassifications on 61 items — not decisive separation.
- **ARI is nominally the best** on every headline number (highest AUC, fewest errors, highest F1),
  but its edge over FK is 2 fewer errors — inside the noise for this sample size.
- **Dale-Chall is the worst separator, by a wide margin** (AUC 0.660, 19 errors). Its familiar-word
  list flags many authored grade-3 passages as hard (they score DC 6.5-7+), so it *mislabels the
  positives*. The hypothesis that DC would catch FK's "short-but-hard-word" blind spot is **not
  supported on this set** — DC underperforms FK badly.

## 2. Combination test: FK ≤ 3.0 AND Dale-Chall ≤ X (the project's current gate style)

Baseline **FK ≤ 3.0 alone**: errors = 9 (FP = 4, FN = 5), precision = 0.889, recall = 0.865, F1 = 0.877.

| Second gate | Errors | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| *(none — FK ≤ 3.0 only)* | **9** | 4 | 5 | 0.889 | 0.865 | **0.877** |
| + DC ≤ 6.5 | 28 | 3 | 25 | 0.800 | 0.324 | 0.462 |
| + DC ≤ 7.0 | 21 | 3 | 18 | 0.864 | 0.514 | 0.644 |
| + DC ≤ 7.5 | 20 | — | — | (recall collapses further) | | |

Adding Dale-Chall as an AND gate **strictly hurts**: it buys at most 1 fewer false positive while
destroying recall (FN jumps from 5 to 18-25), because genuine grade-3 passages routinely exceed any
DC threshold low enough to reject the adult text. **The combination is worse than FK alone at every
operating point.** No AND-combination of FK+DC beats FK on this set.

## 3. Recommendation

**Keep Flesch-Kincaid as the primary gate; do NOT add Dale-Chall.** Dale-Chall is the weakest
formula tested here (AUC 0.66) and adding it as a second gate only wrecks recall, so the specific
hypothesis motivating this eval is refuted. The three formulas that *nominally* edge out FK — ARI,
Coleman-Liau, SMOG — do so by only 1-2 misclassifications on 61 passages, well within sampling
noise, and all four sit in the same AUC band (0.865-0.892); none is a decisive, DOK-worthy
improvement that would justify swapping a working baseline. If a single change were worth making, the
best-supported one is a **small FK threshold nudge** (FK ≤ 3.0 gives 9 errors with balanced FP/FN;
the error-minimising cut is ≈3.75 with recall 1.0 at the cost of 9 false positives) — a tuning
decision, not a formula change. **Net result: nothing clearly beats FK, which is itself the useful
finding — the character-based formulas are interchangeable substitutes at best, and Dale-Chall is a
confirmed dead end for grade-3 detection on this corpus.**

## Combinations + ML

Follow-up: do multi-metric combinations or a lightweight ML model beat FK-alone (9 errors, F1 0.877/0.892)?

**Baseline for this section:** FK-alone at its best single threshold (FK ≤ 3.75) = **9 errors, F1 0.892**.

### (1a) FK AND-gated with each other metric
Both thresholds jointly optimised for fewest errors *on the full 61* — so these are **training-fit
(optimistic) numbers, not cross-validated**; read them as an upper bound, not an estimate of held-out
performance.

| Rule (thresholds fitted on all 61) | Errors | FP | FN | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| **FK-alone** (FK ≤ 3.75) — baseline | 9 | 9 | 0 | 0.852 | 0.804 | 1.000 | 0.892 |
| FK ≤ 3.75 AND ARI ≤ 4.45 | **6** | 6 | 0 | 0.902 | 0.860 | 1.000 | **0.925** |
| FK ≤ 3.75 AND SMOG ≤ 6.81 | 6 | 5 | 1 | 0.902 | 0.878 | 0.973 | 0.923 |
| FK ≤ 3.75 AND Coleman-Liau ≤ 5.95 | 7 | 7 | 0 | 0.885 | 0.841 | 1.000 | 0.914 |
| FK ≤ 3.75 AND Dale-Chall ≤ 9.06 | 9 | 9 | 0 | 0.852 | 0.804 | 1.000 | 0.892 |

The best AND-pairings (FK+ARI, FK+SMOG) shave 3 training-fit errors off FK-alone (9→6) by trimming a
few false positives. FK+Dale-Chall again shows **no gain**: the optimiser pushes DC's cut so high
(≤9.06) that the DC gate never fires — i.e. the fitted "best" DC combo is just FK-alone. Consistent
with §2: DC carries no useful signal here.

### (1b) / (2a) Fitted logistic regression over standardized metrics — LEAVE-ONE-OUT CV
This is the tractable **ML option**. n=61 is tiny, so numbers below are **leave-one-out
cross-validated** (61 refits, standardization fit on the training fold only) — no training-fit
leakage. L2-regularized, threshold 0.5.

| Model | CV Errors | FP | FN | CV Accuracy | CV Precision | CV Recall | CV F1 |
|---|---|---|---|---|---|---|---|
| **FK-alone** (reference) | 9 | 9 | 0 | 0.852 | 0.804 | 1.000 | 0.892 |
| LogReg {FK, ARI} | **8** | 6 | 2 | 0.869 | 0.854 | 0.946 | 0.897 |
| LogReg {FK, ARI, CL, SMOG, DC} | 12 | 7 | 5 | 0.803 | 0.821 | 0.865 | 0.842 |

Under honest cross-validation the small 2-feature model {FK, ARI} lands at 8 errors — **one fewer than
FK-alone, inside sampling noise on 61 items**. The full 5-feature model is *worse* (12 errors): with
only 61 rows it overfits, and the noisy/anti-correlated Dale-Chall feature drags it down. This is the
key methodological point — the AND-combo's 6-error figure is training-fit; the moment you cross-validate
a comparable multi-metric model, the apparent gain largely evaporates.

### (2b) Pretrained ML readability scorer — not run, deliberately
A pretrained transformer readability scorer (e.g. a HuggingFace grade-level regressor) was **not
attempted** within the time-box: it requires a network download and loading a model into an already
tight ~2-3 GB free-RAM budget with no GPU, for a one-off 61-passage scoring — cost far exceeds value
for a bounded eval, and the deterministic formulas already answer the question. The one model-based
signal we *do* have on this exact set is the **gpt-4o LLM-as-judge**, and it was **badly miscalibrated**:
it rated 23 of 24 adult-level FK-10+ passages as "easy for an 8-year-old." So the available
model-as-judge evidence points the wrong way, reinforcing the decision to stay with deterministic
formulas rather than invest in a heavier ML scorer.

### Updated verdict
Under proper leave-one-out cross-validation, **no combination or ML model beats FK-alone by a
meaningful margin** — the best (LogReg {FK, ARI}) is one error better on 61 passages, and the fuller
5-metric model overfits and does worse. The training-fit FK+ARI / FK+SMOG AND-gates look better (6
errors) but that edge is optimistic and does not survive cross-validation, so it is not a sound basis
for changing the gate. This is a clean **null result**: FK-alone remains the right, simplest choice;
if any single enhancement were ever pursued it would be a light FK+ARI touch, but the data does not
justify the added complexity, and Dale-Chall is confirmed useless (and the LLM-judge miscalibrated)
for grade-3 detection on this corpus.

---
*Reproduce: `.venv/Scripts/python.exe eval/run_metric_comparison.py` and
`.venv/Scripts/python.exe eval/run_combinations.py` (raw numbers in `eval/_metric_results.json`,
`eval/_combination_results.json`). textstat 0.7.x; functions `flesch_kincaid_grade`,
`dale_chall_readability_score`, `coleman_liau_index`, `automated_readability_index`, `smog_index`.
Logistic regression hand-rolled (no sklearn available), LOO-CV.*
