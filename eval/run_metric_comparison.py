import json, textstat

PATH = "data/sample/fk_eval_drafts_37.jsonl"
rows = []
with open(PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

# Signals: lower = easier for all these grade formulas (Dale-Chall too)
SIGNALS = {
    "Flesch-Kincaid": textstat.flesch_kincaid_grade,
    "Dale-Chall":     textstat.dale_chall_readability_score,
    "Coleman-Liau":   textstat.coleman_liau_index,
    "ARI":            textstat.automated_readability_index,
    "SMOG":           textstat.smog_index,
}

for r in rows:
    t = r["explanation"]
    r["sig"] = {name: fn(t) for name, fn in SIGNALS.items()}

n = len(rows)
pos = [r for r in rows if r["label_good"]]      # grade-3 (True)
neg = [r for r in rows if not r["label_good"]]  # not grade-3 (False)
P, N = len(pos), len(neg)
print(f"Total={n}  positives(grade3)={P}  negatives(not)={N}")

def auc(scores_pos, scores_neg):
    # Mann-Whitney: prob a random negative scores HIGHER than a random positive
    # (positives should have LOWER grade). AUC=1 => perfect separation with lower=positive.
    c = 0.0
    tot = len(scores_pos) * len(scores_neg)
    for sp in scores_pos:
        for sn in scores_neg:
            if sn > sp:
                c += 1
            elif sn == sp:
                c += 0.5
    return c / tot

def best_threshold(name):
    # predict positive (grade3) if signal <= thr
    vals = sorted(set(r["sig"][name] for r in rows))
    cands = []
    prev = None
    for v in vals:
        if prev is None:
            cands.append(v - 0.01)
        else:
            cands.append((prev + v) / 2)
        prev = v
    cands.append(vals[-1] + 0.01)
    best = None
    for thr in cands:
        tp = sum(1 for r in pos if r["sig"][name] <= thr)
        fn = P - tp
        fp = sum(1 for r in neg if r["sig"][name] <= thr)
        tn = N - fp
        errors = fn + fp
        acc = (tp + tn) / n
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / P if P else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rec_info = (errors, thr, tp, fn, fp, tn, acc, prec, rec, f1)
        if best is None or (errors < best[0]) or (errors == best[0] and f1 > best[9]):
            best = rec_info
    return best

print("\n=== Single-signal separation (predict grade3 if signal <= threshold) ===")
results = {}
for name in SIGNALS:
    a = auc([r["sig"][name] for r in pos], [r["sig"][name] for r in neg])
    errors, thr, tp, fn, fp, tn, acc, prec, rec, f1 = best_threshold(name)
    results[name] = dict(auc=a, thr=thr, errors=errors, tp=tp, fn=fn, fp=fp, tn=tn,
                         acc=acc, prec=prec, rec=rec, f1=f1)
    print(f"{name:15s} AUC={a:.3f}  best_thr<={thr:6.2f}  errors={errors:2d} "
          f"(FP={fp},FN={fn})  acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} F1={f1:.3f}")

# Fixed FK<=3.0 baseline (project's current whole-passage gate)
def eval_pred(predfn):
    tp = sum(1 for r in pos if predfn(r))
    fn = P - tp
    fp = sum(1 for r in neg if predfn(r))
    tn = N - fp
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / P
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, acc=acc, prec=prec, rec=rec, f1=f1, errors=fn+fp)

print("\n=== Fixed project gate FK<=3.0 vs FK+Dale-Chall combination ===")
fk_only = eval_pred(lambda r: r["sig"]["Flesch-Kincaid"] <= 3.0)
print("FK<=3.0 alone:", fk_only)

# Try adding Dale-Chall as second gate at a few thresholds
for dc in [4.9, 5.5, 6.0, 6.5, 7.0]:
    combo = eval_pred(lambda r, dc=dc: r["sig"]["Flesch-Kincaid"] <= 3.0 and r["sig"]["Dale-Chall"] <= dc)
    print(f"FK<=3.0 AND DaleChall<={dc}: {combo}")

# Also: does the FK best threshold differ from 3.0? show FK errors at fixed 3.0
print("\nFK AUC and error curve near 3.0:")
for thr in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    e = eval_pred(lambda r, thr=thr: r["sig"]["Flesch-Kincaid"] <= thr)
    print(f"  FK<={thr}: errors={e['errors']} FP={e['fp']} FN={e['fn']} prec={e['prec']:.3f} rec={e['rec']:.3f} F1={e['f1']:.3f}")

# Dale-Chall alone across thresholds
print("\nDale-Chall error curve:")
for thr in [4.9, 5.5, 6.0, 6.5, 7.0, 7.5]:
    e = eval_pred(lambda r, thr=thr: r["sig"]["Dale-Chall"] <= thr)
    print(f"  DC<={thr}: errors={e['errors']} FP={e['fp']} FN={e['fn']} prec={e['prec']:.3f} rec={e['rec']:.3f} F1={e['f1']:.3f}")

import json as _j
summary = dict(n=n, P=P, N=N, results=results, fk_only=fk_only)
with open("eval/_metric_results.json", "w") as f:
    _j.dump(summary, f, indent=2)
print("\nwrote eval/_metric_results.json")
