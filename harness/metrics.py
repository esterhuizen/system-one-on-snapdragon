#!/usr/bin/env python3
"""metrics.py --items ITEMS.jsonl --runs RUN_DIR... [--reference laya-cpu] [--out metrics.json]"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score
def pvec(ans, q):
    if not ans: return None
    if q["type"] == "noul": p = float(ans["noul"]); return np.array([1 - p, p])
    if q["type"] == "choice": return np.array([float(ans["probabilities"].get(k, np.nan)) for k in list(q["criteria"])])
    return np.array([float(ans["probabilities"].get(str(i), np.nan)) for i in range(len(q["criteria"]))])
def gidx(g, q):
    if g is None: return None
    return int(bool(g)) if q["type"] == "noul" else (list(q["criteria"]).index(g) if q["type"] == "choice" else int(g))
def load(items_path, runs):
    items = {it["id"]: it for it in map(json.loads, open(items_path, encoding="utf-8"))}; rows = []
    for d in runs:
        for fn in glob.glob(os.path.join(d, "*.jsonl")):
            for rec in map(json.loads, open(fn, encoding="utf-8")):
                it = items[rec["item_id"]]; base = {k: rec.get(k) for k in ("target", "rep", "item_id", "status", "attempt", "e2e_ms", "server_ms", "n_questions", "routing_model")}
                base["input_tokens"] = (rec.get("usage") or {}).get("input_tokens")
                for qid, q in it["questions"].items():
                    rows.append({**base, "qid": qid, "qtype": q["type"], "gold": gidx((it.get("gold") or {}).get(qid), q),
                                 "p": pvec((rec.get("answers") or {}).get(qid), q) if rec["status"] == 200 else None})
    return pd.DataFrame(rows)
def ece(conf, corr, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)
def quality(df):
    out, ok = {}, df[df.p.notna() & df.gold.notna()]
    for (t, qt), g in ok.groupby(["target", "qtype"]):
        P, y = list(g.p), g.gold.astype(int).to_numpy(); pred = np.array([int(np.nanargmax(p)) for p in P]); conf = np.array([np.nanmax(p) for p in P])
        r = {"n": len(g), "acc": float((pred == y).mean()), "brier": float(np.mean([np.sum((p - np.eye(len(p))[i]) ** 2) for p, i in zip(P, y)])),
             "nll": float(-np.mean(np.log(np.clip([p[i] for p, i in zip(P, y)], 1e-12, 1)))), "ece": ece(conf, (pred == y).astype(float))}
        if qt == "noul" and len(set(y)) == 2: r["auroc"] = float(roc_auc_score(y, [p[1] for p in P]))
        if qt == "score":
            es = np.array([np.dot(np.arange(len(p)), p) for p in P])
            r.update(mae=float(np.abs(es - y).mean()), spearman=float(spearmanr(es, y).statistic), qwk=float(cohen_kappa_score(y, pred, weights="quadratic")))
        out[f"{t}/{qt}"] = r
    return out
def parity(df, ref):
    f = df[(df.rep == 0) & df.p.notna()].set_index(["item_id", "qid"]); R, out = f[f.target == ref][["p"]], {}
    for t in f.target.unique():
        if t == ref: continue
        j = R.join(f[f.target == t][["p"]], lsuffix="_r", rsuffix="_t", how="inner")
        if j.empty: continue
        out[f"{t} vs {ref}"] = {"n": len(j), "argmax_agree": float(np.mean([np.nanargmax(a) == np.nanargmax(b) for a, b in zip(j.p_r, j.p_t)])),
                                "max_abs_dp": float(max(np.nanmax(np.abs(a - b)) for a, b in zip(j.p_r, j.p_t))),
                                "mean_tv": float(np.mean([0.5 * np.nansum(np.abs(a - b)) for a, b in zip(j.p_r, j.p_t)]))}
    return out
def paired(df, a, b, n_boot=2000):
    g = df[df.p.notna() & df.gold.notna() & (df.rep == 0)]
    c = {t: g[g.target == t].set_index(["item_id", "qid"]).apply(lambda r: int(np.nanargmax(r.p) == r.gold), axis=1) for t in (a, b)}
    j = pd.concat(c, axis=1, join="inner").dropna()
    if j.empty: return None
    x, y = j[a].to_numpy(), j[b].to_numpy(); n01, n10 = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    it = j.index.get_level_values(0).to_numpy(); u = np.unique(it); idx = {k: np.where(it == k)[0] for k in u}; rng = np.random.default_rng(0)
    d = [x[p].mean() - y[p].mean() for p in (np.concatenate([idx[k] for k in rng.choice(u, len(u))]) for _ in range(n_boot))]
    return {"n": len(j), f"acc_{a}": float(x.mean()), f"acc_{b}": float(y.mean()), "diff_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "mcnemar_p": float(binomtest(min(n01, n10), n01 + n10).pvalue) if n01 + n10 else 1.0}
def latency(df):
    c = df.drop_duplicates(["target", "rep", "item_id"]); c = c[(c.status == 200) & (c.attempt == 0)]; out = {}
    for (t, nq), g in c.groupby(["target", "n_questions"]):
        e, s = g.e2e_ms.astype(float), pd.to_numeric(g.server_ms, errors="coerce")
        out[f"{t}/q{nq}"] = {"n": len(g), "e2e_p50": e.quantile(.5), "e2e_p95": e.quantile(.95), "e2e_p99": e.quantile(.99),
                             "server_p50": float(s.quantile(.5)) if s.notna().any() else None}
    return out
def extras(df):
    c = df.drop_duplicates(["target", "rep", "item_id"]); tok = c[c.target.str.startswith("jev") & (c.status == 200)].input_tokens.fillna(0).sum()
    stab = {t: float((g.assign(a=[int(np.nanargmax(p)) for p in g.p]).groupby(["item_id", "qid"]).a.nunique() == 1).mean()) for t, g in df[df.p.notna()].groupby("target")}
    lt = c[c.target.str.startswith("laya") & c.input_tokens.notna()].copy()
    lt["lim"] = np.where(lt.target.str.endswith("-td"), 1000, 500)   # typed-decisions max_len 1024, English root 512
    return {"jev_input_tokens": int(tok), "jev_cost_usd": float(tok) * 0.042e-6, "repeat_argmax_stability": stab,
            "status_counts": {t: g.status.value_counts().to_dict() for t, g in c.groupby("target")},
            "laya_items_likely_truncated": sorted(lt[(lt.input_tokens / lt.n_questions) >= lt.lim].item_id.unique().tolist())[:50]}
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--items", required=True); ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--reference", default="laya-cpu"); ap.add_argument("--out"); a = ap.parse_args(); df = load(a.items, a.runs)
    tg = sorted(df.target.unique()); res = {"parity": parity(df, a.reference), "quality": quality(df), "latency": latency(df), "extras": extras(df),
           "paired_vs_jev": {t: paired(df, t, "jev") for t in tg if t != "jev"} if "jev" in tg else {}}
    s = json.dumps(res, indent=1, default=float); print(s)
    if a.out: open(a.out, "w").write(s)
