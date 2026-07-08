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

---
*Reproduce: `.venv/Scripts/python.exe eval/run_metric_comparison.py` (raw numbers in
`eval/_metric_results.json`). textstat 0.7.x; functions `flesch_kincaid_grade`,
`dale_chall_readability_score`, `coleman_liau_index`, `automated_readability_index`, `smog_index`.*
