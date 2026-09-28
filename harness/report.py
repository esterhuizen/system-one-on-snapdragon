"""Comparison report for a suite run: quality vs consensus gold, Jev<->Laya agreement, parity, latency.

python -X utf8 report.py --items ITEMS.jsonl --run RUN_DIR [--reference laya-cpu] [--out report.json]

Every quality metric is computed on rep 0, per (tier, target, qtype); latency uses all reps.
Probabilities are clipped to [0.005, 1] for log-loss because Jev rounds to 2 decimals (0.0 would be -inf).
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest, spearmanr

EPS = 0.005
ap = argparse.ArgumentParser(); ap.add_argument("--items", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--reference", default="laya-cpu"); ap.add_argument("--out"); a = ap.parse_args()
items = {it["id"]: it for it in map(json.loads, open(a.items, encoding="utf-8"))}

def pvec(ans, q):
    if not ans: return None
    if q["type"] == "noul": p = float(ans["noul"]); return np.array([1 - p, p])
    keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
    v = np.array([float((ans.get("probabilities") or {}).get(k, 0.0)) for k in keys])
    return v / v.sum() if v.sum() > 0 else None

def gidx(g, q):
    if g is None: return None
    if q["type"] == "noul": return int(g is True or str(g).lower() == "true")
    if q["type"] == "choice": return list(q["criteria"]).index(g)
    return int(g)

rows, calls = [], []
for fn in glob.glob(os.path.join(a.run, "*", "*.jsonl")):
    for rec in map(json.loads, open(fn, encoding="utf-8")):
        it = items[rec["item_id"]]
        calls.append({k: rec.get(k) for k in ("target", "rep", "item_id", "status", "attempt", "e2e_ms", "server_ms", "n_questions")}
                     | {"tier": it["tier"], "domain": it["domain"], "input_tokens": (rec.get("usage") or {}).get("input_tokens")})
        for qid, q in it["questions"].items():
            m = (it.get("meta") or {}).get(qid, {})
            p = pvec((rec.get("answers") or {}).get(qid), q) if rec["status"] == 200 else None
            rows.append({"target": rec["target"], "rep": rec["rep"], "item_id": rec["item_id"], "qid": qid, "tier": it["tier"],
                         "domain": it["domain"], "qtype": q["type"], "k": len(q.get("criteria") or [0, 1]) if q["type"] != "noul" else 2,
                         "difficulty": m.get("difficulty"), "evidence": m.get("evidence_position"), "consensus": m.get("consensus"),
                         "gold": gidx((it.get("gold") or {}).get(qid), q), "status": rec["status"], "p": p,
                         "score": ((rec.get("answers") or {}).get(qid) or {}).get("score")})
df, cf = pd.DataFrame(rows), pd.DataFrame(calls)
targets = sorted(df.target.unique(), key=lambda t: ["jev", "laya-npu", "laya-gpu", "laya-ortcpu", "laya-cpu"].index(t) if t in ["jev", "laya-npu", "laya-gpu", "laya-ortcpu", "laya-cpu"] else 9)
r0 = df[df.rep == 0]

def ece(conf, corr, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)

def quality(g):
    g = g[g.p.notna() & g.gold.notna()]
    if g.empty: return None
    P, y = list(g.p), g.gold.astype(int).to_numpy()
    pred = np.array([int(np.argmax(p)) for p in P]); conf = np.array([float(np.max(p)) for p in P]); corr = (pred == y).astype(float)
    r = {"n": int(len(g)), "acc": float(corr.mean()),
         "brier": float(np.mean([np.sum((p - np.eye(len(p))[i]) ** 2) for p, i in zip(P, y)])),
         "nll": float(-np.mean([np.log(max(EPS, p[i])) for p, i in zip(P, y)])), "ece": ece(conf, corr), "mean_conf": float(conf.mean())}
    s = g[g.qtype == "score"]
    if len(s) >= 3:
        es = np.array([float(np.dot(np.arange(len(p)), p)) for p in s.p]); ys = s.gold.astype(float).to_numpy()
        r |= {"score_mae": float(np.abs(es - ys).mean()), "score_spearman": float(spearmanr(es, ys).statistic)}
    return r

def by(keys, frame=r0):
    out = {}
    for k, g in frame.groupby(keys):
        q = quality(g)
        if q: out["/".join(map(str, k if isinstance(k, tuple) else (k,)))] = q
    return out

def agree(x, y, frame=r0):
    A = frame[(frame.target == x) & frame.p.notna()].set_index(["item_id", "qid"])
    B = frame[(frame.target == y) & frame.p.notna()].set_index(["item_id", "qid"])
    j = A[["p", "tier", "qtype"]].join(B[["p"]], rsuffix="_y", how="inner")
    if j.empty: return {}
    j["agree"] = [int(np.argmax(p) == np.argmax(q)) for p, q in zip(j.p, j.p_y)]
    j["tv"] = [0.5 * float(np.abs(p - q).sum()) for p, q in zip(j.p, j.p_y)]
    j["maxdp"] = [float(np.abs(p - q).max()) for p, q in zip(j.p, j.p_y)]
    out = {}
    for key, g in [("all", j)] + [(f"{t}", g) for t, g in j.groupby("tier")] + [(f"{t}/{qt}", g) for (t, qt), g in j.groupby(["tier", "qtype"])]:
        out[key] = {"n": int(len(g)), "argmax_agree": float(g.agree.mean()), "mean_tv": float(g.tv.mean()), "max_abs_dp": float(g.maxdp.max())}
    return out

def paired(x, y="jev"):
    out = {}
    for tier, g in r0[r0.gold.notna() & r0.p.notna()].groupby("tier"):
        c = {t: g[g.target == t].set_index(["item_id", "qid"]).apply(lambda r: int(np.argmax(r.p) == r.gold), axis=1) for t in (x, y)}
        if any(v.empty for v in c.values()): continue
        j = pd.concat(c, axis=1, join="inner").dropna()
        u, v = j[x].to_numpy(), j[y].to_numpy(); n01, n10 = int(((u == 0) & (v == 1)).sum()), int(((u == 1) & (v == 0)).sum())
        ids = j.index.get_level_values(0).to_numpy(); uniq = np.unique(ids); idx = {k: np.where(ids == k)[0] for k in uniq}; rng = np.random.default_rng(0)
        d = [u[s].mean() - v[s].mean() for s in (np.concatenate([idx[k] for k in rng.choice(uniq, len(uniq))]) for _ in range(4000))]
        out[tier] = {"n": int(len(j)), f"acc_{x}": float(u.mean()), f"acc_{y}": float(v.mean()), "diff": float(u.mean() - v.mean()),
                     "diff_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], f"{x}_only": n10, f"{y}_only": n01,
                     "mcnemar_p": float(binomtest(min(n01, n10), n01 + n10).pvalue) if n01 + n10 else 1.0}
    return out

def latency():
    c = cf[(cf.status == 200) & (cf.attempt == 0)].copy(); c["per_q"] = c.e2e_ms / c.n_questions; out = {}
    for (t, tier), g in c.groupby(["target", "tier"]):
        s = pd.to_numeric(g.server_ms, errors="coerce")
        out[f"{t}/{tier}"] = {"calls": int(len(g)), "e2e_p50": float(g.e2e_ms.median()), "e2e_p95": float(g.e2e_ms.quantile(.95)),
                               "per_q_p50": float(g.per_q.median()), "server_p50": float(s.median()) if s.notna().any() else None,
                               "mean_questions": float(g.n_questions.mean())}
    return out

meta_path = next(iter(glob.glob(os.path.join(a.run, "jev", "meta.json"))), None)
jev_rtt = None
if meta_path:
    pings = json.load(open(meta_path)).get("pings") or []
    jev_rtt = next((p["rtt_p50_ms"] for p in pings if p["target"] == "jev"), None)
cj = cf[(cf.target == "jev") & (cf.status == 200)]
stab = {t: float((g[g.p.notna()].assign(a=[int(np.argmax(p)) for p in g[g.p.notna()].p]).groupby(["item_id", "qid"]).a.nunique() == 1).mean())
        for t, g in df.groupby("target") if g.rep.nunique() > 1}
gold_stats = {"questions": int(sum(len(it["questions"]) for it in items.values())),
              "with_gold": int(sum(len(it.get("gold") or {}) for it in items.values())),
              "consensus": pd.Series([m.get("consensus") for it in items.values() for m in (it.get("meta") or {}).values()]).value_counts().to_dict(),
              "items_per_tier": pd.Series([it["tier"] for it in items.values()]).value_counts().to_dict()}
res = {
    "targets": targets, "gold": gold_stats,
    "status": {t: {tier: g.status.value_counts().to_dict() for tier, g in cf[cf.target == t].groupby("tier")} for t in targets},
    "quality_tier": by(["tier", "target"]), "quality_tier_qtype": by(["tier", "target", "qtype"]),
    "quality_difficulty": by(["tier", "target", "difficulty"]), "quality_domain": by(["target", "domain"]),
    "quality_evidence_t512": by(["target", "evidence"], r0[r0.tier == "t512"]),
    "agreement_vs_jev": {t: agree(t, "jev") for t in targets if t != "jev"},
    "parity_vs_reference": {t: agree(t, a.reference) for t in targets if t not in ("jev", a.reference)},
    "paired_vs_jev": {t: paired(t) for t in targets if t != "jev"},
    "latency": latency(), "jev_ping_rtt_p50_ms": jev_rtt, "repeat_argmax_stability": stab,
    "jev_cost": {"input_tokens": int(cj.input_tokens.fillna(0).sum()), "usd": float(cj.input_tokens.fillna(0).sum() * 0.042e-6)},
}
s = json.dumps(res, indent=1, default=float)
if a.out: open(a.out, "w", encoding="utf-8").write(s)

# ---- console summary ----
print(f"gold: {gold_stats}")
for tier in sorted(r0.tier.unique()):
    print(f"\n== {tier}   acc / brier / ece / mean_conf  (n questions with consensus gold)")
    for t in targets:
        q = res["quality_tier"].get(f"{tier}/{t}")
        lat = res["latency"].get(f"{t}/{tier}")
        if q: print(f"  {t:<12} acc {q['acc']:.3f}  brier {q['brier']:.3f}  ece {q['ece']:.3f}  conf {q['mean_conf']:.3f}  n={q['n']}"
                    + (f"   | call p50 {lat['e2e_p50']:.0f} ms, per-q {lat['per_q_p50']:.0f} ms" if lat else ""))
    for t in targets:
        ag = res["agreement_vs_jev"].get(t, {}).get(tier)
        if ag: print(f"  agree {t:<12} vs jev: argmax {ag['argmax_agree']:.3f}  mean TV {ag['mean_tv']:.3f}  n={ag['n']}")
print("\nparity vs", a.reference, {t: {k: v for k, v in (d.get('all') or {}).items()} for t, d in res["parity_vs_reference"].items()})
print("jev cost", res["jev_cost"], "| jev ping p50", jev_rtt)
