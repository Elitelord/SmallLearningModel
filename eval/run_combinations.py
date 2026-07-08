import json, math, textstat

PATH = "data/sample/fk_eval_drafts_37.jsonl"
rows = []
with open(PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

SIGNALS = {
    "FK":   textstat.flesch_kincaid_grade,
    "ARI":  textstat.automated_readability_index,
    "CL":   textstat.coleman_liau_index,
    "SMOG": textstat.smog_index,
    "DC":   textstat.dale_chall_readability_score,
}
for r in rows:
    t = r["explanation"]
    r["sig"] = {k: fn(t) for k, fn in SIGNALS.items()}
    r["y"] = 1 if r["label_good"] else 0

n = len(rows)
P = sum(r["y"] for r in rows)
N = n - P

def metrics(pred):
    tp = sum(1 for r, p in zip(rows, pred) if p and r["y"])
    fp = sum(1 for r, p in zip(rows, pred) if p and not r["y"])
    fn = sum(1 for r, p in zip(rows, pred) if not p and r["y"])
    tn = n - tp - fp - fn
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(errors=fp + fn, fp=fp, fn=fn, acc=acc, prec=prec, rec=rec, f1=f1)

def cands(name):
    vals = sorted(set(r["sig"][name] for r in rows))
    c = [vals[0] - 0.01]
    for i in range(1, len(vals)):
        c.append((vals[i-1] + vals[i]) / 2)
    c.append(vals[-1] + 0.01)
    return c

# ---- FK-alone baseline (best single threshold) ----
best_fk = None
for thr in cands("FK"):
    m = metrics([r["sig"]["FK"] <= thr for r in rows])
    if best_fk is None or m["errors"] < best_fk[0]["errors"] or (m["errors"] == best_fk[0]["errors"] and m["f1"] > best_fk[0]["f1"]):
        best_fk = (m, thr)
print("FK-alone baseline: thr<=%.2f %s" % (best_fk[1], best_fk[0]))

# ---- (1a) FK AND partner, jointly optimized thresholds ----
print("\n=== FK AND partner (both thresholds jointly optimized for fewest errors) ===")
combo_rows = []
for partner in ["ARI", "CL", "SMOG", "DC"]:
    best = None
    for ft in cands("FK"):
        for pt in cands(partner):
            m = metrics([(r["sig"]["FK"] <= ft and r["sig"][partner] <= pt) for r in rows])
            if best is None or m["errors"] < best[0]["errors"] or (m["errors"] == best[0]["errors"] and m["f1"] > best[0]["f1"]):
                best = (m, ft, pt)
    m, ft, pt = best
    combo_rows.append((f"FK<= {ft:.2f} AND {partner}<= {pt:.2f}", m))
    print(f"FK+{partner:4s}: FK<={ft:5.2f} {partner}<={pt:6.2f} -> {m}")

# ---- (1b) Logistic regression with leave-one-out CV ----
def standardize(cols):
    means, sds = {}, {}
    for k in cols:
        xs = [r["sig"][k] for r in rows]
        mu = sum(xs) / len(xs)
        var = sum((x - mu) ** 2 for x in xs) / len(xs)
        means[k] = mu
        sds[k] = math.sqrt(var) if var > 0 else 1.0
    return means, sds

def fit_logreg(train, cols, means, sds, iters=4000, lr=0.1, l2=1.0):
    # features standardized using provided means/sds (from train only)
    w = [0.0] * (len(cols) + 1)  # last is bias
    m = len(train)
    for _ in range(iters):
        grad = [0.0] * len(w)
        for r in train:
            x = [(r["sig"][k] - means[k]) / sds[k] for k in cols] + [1.0]
            z = sum(wi * xi for wi, xi in zip(w, x))
            p = 1 / (1 + math.exp(-z))
            err = p - r["y"]
            for j in range(len(w)):
                grad[j] += err * x[j]
        for j in range(len(w)):
            reg = l2 * w[j] if j < len(cols) else 0.0  # no reg on bias
            w[j] -= lr * (grad[j] / m + reg / m)
    return w, means, sds

def loo_cv(cols):
    preds = []
    for i in range(n):
        train = [rows[j] for j in range(n) if j != i]
        # standardize on train only
        means, sds = {}, {}
        for k in cols:
            xs = [r["sig"][k] for r in train]
            mu = sum(xs) / len(xs)
            var = sum((x - mu) ** 2 for x in xs) / len(xs)
            means[k] = mu
            sds[k] = math.sqrt(var) if var > 0 else 1.0
        w, _, _ = fit_logreg(train, cols, means, sds)
        r = rows[i]
        x = [(r["sig"][k] - means[k]) / sds[k] for k in cols] + [1.0]
        z = sum(wi * xi for wi, xi in zip(w, x))
        p = 1 / (1 + math.exp(-z))
        preds.append(1 if p >= 0.5 else 0)
    return metrics([bool(p) for p in preds])

print("\n=== Logistic regression, LEAVE-ONE-OUT CV (no training-fit leakage) ===")
lr_all = loo_cv(["FK", "ARI", "CL", "SMOG", "DC"])
lr_fkari = loo_cv(["FK", "ARI"])
print("LogReg {FK,ARI,CL,SMOG,DC} LOO-CV:", lr_all)
print("LogReg {FK,ARI}         LOO-CV:", lr_fkari)

out = {
    "baseline_fk": {"thr": best_fk[1], **best_fk[0]},
    "and_combos": [{"rule": r, **m} for r, m in combo_rows],
    "logreg_all_loocv": lr_all,
    "logreg_fkari_loocv": lr_fkari,
}
with open("eval/_combination_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote eval/_combination_results.json")
