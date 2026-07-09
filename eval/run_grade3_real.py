"""Score the INDEPENDENT grade-3 ground-truth set (CLEAR) with every metric.

Answers the two questions the FK-tuned eval could not:
  Q1 (calibration): where does GENUINE grade-3 material land on each metric?
                    i.e. is the project's "grade 3 = FK 1.5-3.0" band even right?
  Q2 (which metric): which single metric / combination most accurately scores
                    real grade-3 text AS grade-3 while still rejecting adult text?

Test set: data/sample/grade3_real_clear.jsonl (label_good from CLEAR Lexile band,
formula-independent). Gradient: eval/_grade3_real_gradient.json (all Info bands).

All formulas: lower = easier. FK/ARI/CL/SMOG additionally OUTPUT a grade estimate,
so "scores it as grade 3" = grade output falling in a grade-3 window.

Reproduce: .venv\\Scripts\\python.exe eval/run_grade3_real.py
Raw numbers -> eval/_grade3_real_results.json
"""

import json
import math
import statistics as st
import sys
from pathlib import Path

import textstat

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "sample" / "grade3_real_clear.jsonl"
GRAD = REPO / "eval" / "_grade3_real_gradient.json"
OUT = REPO / "eval" / "_grade3_real_results.json"

SIGNALS = {
    "Flesch-Kincaid": textstat.flesch_kincaid_grade,
    "ARI":            textstat.automated_readability_index,
    "Coleman-Liau":   textstat.coleman_liau_index,
    "SMOG":           textstat.smog_index,
    "Dale-Chall":     textstat.dale_chall_readability_score,
}
# metrics whose output is itself a US grade level (so "scores as grade 3" is meaningful)
GRADE_METRICS = ["Flesch-Kincaid", "ARI", "Coleman-Liau", "SMOG"]
GRADE3_WINDOW = (2.0, 4.0)   # a metric "calls it grade 3" if grade output in [2,4]


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def score_rows(rows):
    for r in rows:
        t = r["text"]
        r["sig"] = {name: fn(t) for name, fn in SIGNALS.items()}
    return rows


def auc(sp, sn):
    # prob a random negative scores HIGHER than a random positive (lower=grade3)
    c = 0.0
    for a in sp:
        for b in sn:
            c += 1.0 if b > a else (0.5 if b == a else 0.0)
    return c / (len(sp) * len(sn))


def best_threshold(name, rows, pos, neg):
    P, N, n = len(pos), len(neg), len(rows)
    vals = sorted(set(r["sig"][name] for r in rows))
    cands = [vals[0] - 0.01]
    for i in range(1, len(vals)):
        cands.append((vals[i - 1] + vals[i]) / 2)
    cands.append(vals[-1] + 0.01)
    best = None
    for thr in cands:
        tp = sum(1 for r in pos if r["sig"][name] <= thr)
        fp = sum(1 for r in neg if r["sig"][name] <= thr)
        fn, tn = P - tp, N - fp
        errors = fn + fp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / P if P else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        info = dict(errors=errors, thr=round(thr, 2), tp=tp, fn=fn, fp=fp, tn=tn,
                    acc=round((tp + tn) / n, 3), prec=round(prec, 3),
                    rec=round(rec, 3), f1=round(f1, 3))
        if best is None or errors < best["errors"] or (errors == best["errors"] and f1 > best["f1"]):
            best = info
    return best


def eval_pred(predfn, pos, neg):
    P, N, n = len(pos), len(neg), len(pos) + len(neg)
    tp = sum(1 for r in pos if predfn(r))
    fp = sum(1 for r in neg if predfn(r))
    fn, tn = P - tp, N - fp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / P if P else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(errors=fn + fp, tp=tp, fn=fn, fp=fp, tn=tn,
                acc=round((tp + tn) / n, 3), prec=round(prec, 3),
                rec=round(rec, 3), f1=round(f1, 3))


def loo_logistic(rows, feats, pos_label=True):
    """Leave-one-out CV logistic regression over standardized features (no sklearn)."""
    X = [[r["sig"][f] for f in feats] for r in rows]
    y = [1 if r["label_good"] == pos_label else 0 for r in rows]
    n, k = len(X), len(feats)
    errors = fp = fn = 0
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        mus = [st.mean(X[j][c] for j in idx) for c in range(k)]
        sds = [st.pstdev([X[j][c] for j in idx]) or 1.0 for c in range(k)]
        Xtr = [[(X[j][c] - mus[c]) / sds[c] for c in range(k)] for j in idx]
        ytr = [y[j] for j in idx]
        w = [0.0] * k
        b = 0.0
        lr, lam = 0.3, 1.0
        for _ in range(300):
            gw = [0.0] * k
            gb = 0.0
            for xr, yr in zip(Xtr, ytr):
                z = b + sum(w[c] * xr[c] for c in range(k))
                p = 1 / (1 + math.exp(-max(-30, min(30, z))))
                d = p - yr
                for c in range(k):
                    gw[c] += d * xr[c]
                gb += d
            m = len(Xtr)
            for c in range(k):
                w[c] -= lr * (gw[c] / m + lam * w[c] / m)
            b -= lr * gb / m
        xt = [(X[i][c] - mus[c]) / sds[c] for c in range(k)]
        z = b + sum(w[c] * xt[c] for c in range(k))
        pred = 1 if z >= 0 else 0
        if pred != y[i]:
            errors += 1
            if y[i] == 0:
                fp += 1
            else:
                fn += 1
    return dict(cv_errors=errors, fp=fp, fn=fn, n=n)


def main():
    rows = score_rows(load(DATA))
    pos = [r for r in rows if r["label_good"]]
    neg = [r for r in rows if not r["label_good"]]
    n, P, N = len(rows), len(pos), len(neg)
    print(f"Independent CLEAR test set: n={n}  real-grade3(pos)={P}  adult(neg)={N}\n")

    out = {"n": n, "P": P, "N": N}

    # -- Q1a: absolute calibration on real grade-3 positives --
    print("=== Q1  Where does REAL grade-3 material land? (positives only) ===")
    print(f"{'metric':16s} {'mean':>7s} {'median':>7s} {'p10':>6s} {'p90':>6s}")
    calib = {}
    for name in SIGNALS:
        xs = sorted(r["sig"][name] for r in pos)
        mean = st.mean(xs)
        med = st.median(xs)
        p10 = xs[max(0, int(0.10 * len(xs)) - 1)]
        p90 = xs[min(len(xs) - 1, int(0.90 * len(xs)))]
        calib[name] = dict(mean=round(mean, 2), median=round(med, 2),
                           p10=round(p10, 2), p90=round(p90, 2))
        print(f"{name:16s} {mean:7.2f} {med:7.2f} {p10:6.2f} {p90:6.2f}")
    out["calibration_on_real_grade3"] = calib
    fk_mean = calib["Flesch-Kincaid"]["mean"]
    print(f"\n  -> project gate targets FK 1.5-3.0; real grade-3 prose averages FK {fk_mean}.")
    print(f"     Fraction of real grade-3 passing FK<=3.0: "
          f"{sum(1 for r in pos if r['sig']['Flesch-Kincaid']<=3.0)/P:.1%}")

    # -- Q1b: per-Lexile-band gradient (mean metric per band) --
    print("\n=== Q1  Per-Lexile-band gradient (mean metric value; Info excerpts) ===")
    grad = json.load(open(GRAD, encoding="utf-8"))
    band_grade = {"300": "~G1-2", "500": "~G2-3", "700": "~G3-4", "900": "~G4-5",
                  "1100": "~G6-8", "1300": "~G9-11", "1500": "~G11-12", "1700": "adult+"}
    print(f"{'band':>5s} {'~grade':>7s} {'n':>4s} " + " ".join(f"{m[:5]:>6s}" for m in SIGNALS))
    gradient_tbl = {}
    for b, brows in grad.items():
        if not brows:
            continue
        brows = score_rows(brows)
        means = {m: round(st.mean(r["sig"][m] for r in brows), 2) for m in SIGNALS}
        gradient_tbl[b] = dict(n=len(brows), means=means)
        print(f"{b:>5s} {band_grade.get(b,'?'):>7s} {len(brows):>4d} "
              + " ".join(f"{means[m]:6.2f}" for m in SIGNALS))
    out["gradient"] = gradient_tbl

    # -- Q2a: single-signal discrimination (grade-3 vs adult), independent labels --
    print("\n=== Q2  Single-signal separation on INDEPENDENT labels ===")
    print(f"{'metric':16s} {'AUC':>6s} {'bestThr<=':>9s} {'err':>4s} {'FP':>3s} {'FN':>3s} {'F1':>6s}")
    single = {}
    for name in SIGNALS:
        a = auc([r["sig"][name] for r in pos], [r["sig"][name] for r in neg])
        bt = best_threshold(name, rows, pos, neg)
        single[name] = dict(auc=round(a, 3), **bt)
        print(f"{name:16s} {a:6.3f} {bt['thr']:9.2f} {bt['errors']:4d} "
              f"{bt['fp']:3d} {bt['fn']:3d} {bt['f1']:6.3f}")
    out["single_signal"] = single

    # -- Q2b: "scores it AS grade 3" — grade output in [2,4] window --
    print(f"\n=== Q2  Does the metric literally score grade-3 text AS grade 3? "
          f"(grade output in [{GRADE3_WINDOW[0]},{GRADE3_WINDOW[1]}]) ===")
    print(f"{'metric':16s} {'pos_in_win%':>11s} {'neg_in_win%':>11s} {'errors':>7s}")
    window = {}
    lo, hi = GRADE3_WINDOW
    for name in GRADE_METRICS:
        pos_in = sum(1 for r in pos if lo <= r["sig"][name] <= hi)
        neg_in = sum(1 for r in neg if lo <= r["sig"][name] <= hi)
        # error = grade-3 scored OUTSIDE window (miss) + adult scored INSIDE (false alarm)
        errors = (P - pos_in) + neg_in
        window[name] = dict(pos_in_window=pos_in, neg_in_window=neg_in,
                            pos_rate=round(pos_in / P, 3), neg_rate=round(neg_in / N, 3),
                            errors=errors)
        print(f"{name:16s} {pos_in/P*100:10.1f}% {neg_in/N*100:10.1f}% {errors:7d}")
    out["grade3_window"] = window

    # -- Q2c: combinations (AND-gates, fitted on all -> optimistic) + LOO logistic --
    print("\n=== Q2  Combinations ===")
    fk_best = single["Flesch-Kincaid"]["thr"]
    base = eval_pred(lambda r: r["sig"]["Flesch-Kincaid"] <= fk_best, pos, neg)
    print(f"FK-alone (FK<={fk_best}): errors={base['errors']} F1={base['f1']}")
    combos = {"FK_alone": dict(thr=fk_best, **base)}
    for other in ["ARI", "Coleman-Liau", "SMOG", "Dale-Chall"]:
        # jointly optimise the second threshold for fewest errors (training-fit)
        best = None
        for r in sorted(rows, key=lambda x: x["sig"][other]):
            thr2 = r["sig"][other]
            res = eval_pred(lambda r, t=thr2: r["sig"]["Flesch-Kincaid"] <= fk_best
                            and r["sig"][other] <= t, pos, neg)
            if best is None or res["errors"] < best[1]["errors"]:
                best = (round(thr2, 2), res)
        combos[f"FK_AND_{other}"] = dict(thr2=best[0], **best[1])
        print(f"FK<={fk_best} AND {other}<={best[0]}: errors={best[1]['errors']} "
              f"F1={best[1]['f1']}  (training-fit)")
    out["combinations_trainfit"] = combos

    print("\n  Leave-one-out CV logistic (honest):")
    cv = {}
    for feats in (["Flesch-Kincaid"], ["Flesch-Kincaid", "ARI"],
                  ["Flesch-Kincaid", "ARI", "SMOG", "Coleman-Liau"],
                  ["Flesch-Kincaid", "ARI", "SMOG", "Coleman-Liau", "Dale-Chall"]):
        res = loo_logistic(rows, feats)
        key = "+".join(f[:2] for f in feats)
        cv[key] = res
        print(f"    {{{', '.join(feats)}}}: CV errors={res['cv_errors']} "
              f"(FP={res['fp']} FN={res['fn']})")
    out["loo_cv_logistic"] = cv

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
