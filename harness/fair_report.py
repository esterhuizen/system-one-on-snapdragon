"""Metrics for the fair public-benchmark suite (results/items/fair/*.jsonl) across run targets.

python -X utf8 fair_report.py --items-dir DIR --run RUN_DIR [--out report.json]
Groups by (part, family, tier) where family = question id for the fixed-schema parts and question type for gate_td.
Every item's `meta[qid].twin_of` marks the A/B-choice twin of a yes/no question.
Rep 0 is used for quality; all reps for Jev run-to-run stability.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

EPS = 0.005
ap = argparse.ArgumentParser(); ap.add_argument("--items-dir", required=True); ap.add_argument("--run", required=True); ap.add_argument("--out")
a = ap.parse_args()
items = {}
EXCLUDE = {"cfpb-0877", "cfpb-0429", "cfpb-0465"}   # consumer text quotes the label field itself (cfpb verifier)
for fn in sorted(glob.glob(os.path.join(a.items_dir, "*.jsonl"))):
    if fn.endswith(".ref.jsonl"): continue
    part = os.path.basename(fn)[:-6]
    for l in open(fn, encoding="utf-8"):
        if l.strip():
            it = json.loads(l); it["_part"] = part
            if it["id"] not in EXCLUDE: items[it["id"]] = it

def yes_key(it, qid):   # A/B twin: the key whose description equals the yes/no question's own "true" description
    q = it["questions"][qid]; base = (it.get("meta") or {}).get(qid, {}).get("twin_of"); nq = it["questions"].get(base) or {}
    true_txt = ((nq.get("criteria") or {}).get("true") or "").strip().lower()
    k = next((k for k, v in q["criteria"].items() if true_txt and str(v).strip().lower() == true_txt), None)
    if k is None and base in (it.get("gold") or {}) and qid in (it.get("gold") or {}):   # fall back to the gold mapping
        other = [x for x in q["criteria"] if x != it["gold"][qid]]
        k = it["gold"][qid] if it["gold"][base] else (other[0] if other else None)
    return k

rows = []
for fn in glob.glob(os.path.join(a.run, "*", "*.jsonl")):
    for rec in map(json.loads, open(fn, encoding="utf-8")):
        it = items.get(rec["item_id"])
        if it is None: continue
        for qid, q in it["questions"].items():
            m = (it.get("meta") or {}).get(qid, {}); g = (it.get("gold") or {}).get(qid)
            ans = (rec.get("answers") or {}).get(qid) if rec["status"] == 200 else None
            fam = q["type"] if it["_part"] == "gate_td" else (m.get("twin_of") + "_ab" if m.get("twin_of") else qid)
            r = {"target": rec["target"], "rep": rec["rep"], "item": rec["item_id"], "qid": qid, "part": it["_part"], "family": fam,
                 "tier": it.get("tier", "?"), "type": q["type"], "twin": bool(m.get("twin_of")), "status": rec["status"],
                 "slice": (it.get("src") or {}).get("slice"), "template": (it.get("src") or {}).get("template_letter"),
                 "soft": m.get("soft_gold"), "e2e_ms": rec.get("e2e_ms"), "nq": rec.get("n_questions")}
            keys = list(q["criteria"]) if q["type"] == "choice" else ([str(i) for i in range(len(q["criteria"]))] if q["type"] == "score" else None)
            if q["type"] == "noul":
                r["gold"] = None if g is None else int(bool(g)); r["p_yes"] = float(ans["noul"]) if ans else None
                r["pred"] = None if ans is None else int(ans["noul"] >= 0.5); r["probs"] = None if ans is None else [1 - ans["noul"], ans["noul"]]
            else:
                p = [float((ans.get("probabilities") or {}).get(k, 0.0)) for k in keys] if ans else None
                r["probs"] = p; r["pred"] = None if p is None else int(np.argmax(p))
                r["gold"] = None if g is None else (keys.index(str(g)) if q["type"] == "choice" else int(g))
                if m.get("twin_of") and ans:
                    yk = yes_key(it, qid); r["p_yes"] = p[keys.index(yk)] if yk in keys else None
                    r["gold_yes"] = None if g is None else int(str(g) == yk)
                if q["type"] == "score" and p: r["exp"] = float(np.dot(np.arange(len(p)), p))
            rows.append(r)
df = pd.DataFrame(rows); r0 = df[df.rep == 0]
targets = [t for t in ["jev", "winnow-npu", "decider-npu", "laya-npu", "laya-gpu", "laya-cpu"] if t in set(df.target)]

def ece(conf, corr, bins=10):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)

def metrics(g):
    g = g[g.probs.notna()]; out = {"n_answered": int(len(g))}
    gg = g[g.gold.notna()]
    if gg.empty: return out
    y = gg.gold.astype(int).to_numpy(); pred = gg.pred.astype(int).to_numpy(); P = list(gg.probs); t = gg.type.iloc[0]
    out |= {"n": int(len(gg)), "acc": float((pred == y).mean())}
    conf = np.array([max(p) for p in P]); out["ece"] = ece(conf, (pred == y).astype(float))
    out["brier"] = float(np.mean([np.sum((np.array(p) - np.eye(len(p))[i]) ** 2) for p, i in zip(P, y)]))
    if t == "choice" and not gg.twin.iloc[0]:
        out["macro_f1"] = float(f1_score(y, pred, average="macro", labels=sorted(set(y))))
    yes = gg.gold_yes.astype(float).to_numpy() if gg.twin.iloc[0] else (y.astype(float) if t == "noul" else None)
    if yes is not None:
        py = gg.p_yes.astype(float).to_numpy(); ph = (py >= 0.5).astype(int)
        out |= {"base_rate_yes": float(yes.mean()), "pred_rate_yes": float(ph.mean())}
        if len(set(yes)) == 2:
            out["balanced_acc"] = float(0.5 * ((ph[yes == 1] == 1).mean() + (ph[yes == 0] == 0).mean()))
            out["auroc"] = float(roc_auc_score(yes, py))
    if t == "score":
        ex = gg.exp.astype(float).to_numpy()
        out |= {"within1": float((np.abs(pred - y) <= 1).mean()), "mae_expected": float(np.abs(ex - y).mean()),
                "within_half": float((np.abs(ex - y) <= 0.5).mean()),   # sysone-bench SST-5 rule: |E[level] - gold| <= 0.5
                "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")) if len(set(y)) > 1 else None,
                "spearman": float(spearmanr(ex, y).statistic) if len(set(y)) > 1 else None,
                "pred_hist": np.bincount(pred, minlength=len(P[0])).tolist(), "gold_hist": np.bincount(y, minlength=len(P[0])).tolist()}
    soft = g[g.soft.notna() & g.p_yes.notna()] if "p_yes" in g else g.iloc[0:0]
    if len(soft) > 10:
        out["soft_brier"] = float(np.mean((soft.p_yes.astype(float) - soft.soft.astype(float)) ** 2))
        out["soft_spearman"] = float(spearmanr(soft.p_yes.astype(float), soft.soft.astype(float)).statistic)
    return out

def primary(t, twin):  # metric used for the paired comparison
    return "balanced_acc" if (t == "noul" or twin) else ("qwk" if t == "score" else "acc")

def paired(g, a_t="jev", b_t="laya-npu", n_boot=2000):
    A = g[(g.target == a_t) & g.gold.notna() & g.probs.notna()].set_index(["item", "qid"])
    B = g[(g.target == b_t) & g.gold.notna() & g.probs.notna()].set_index(["item", "qid"])
    j = A[["gold", "pred", "type", "twin"]].join(B[["pred"]], rsuffix="_b", how="inner")
    if len(j) < 10: return None
    ca, cb = (j.pred == j.gold).astype(float).to_numpy(), (j.pred_b == j.gold).astype(float).to_numpy()
    rng = np.random.default_rng(0); n = len(j)
    d = [ca[s].mean() - cb[s].mean() for s in (rng.integers(0, n, n) for _ in range(n_boot))]
    return {"n": n, "acc_diff": float(ca.mean() - cb.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            f"{a_t}_only": int(((ca == 1) & (cb == 0)).sum()), f"{b_t}_only": int(((ca == 0) & (cb == 1)).sum())}

res = {"groups": {}, "paired": {}, "cfpb_slices": {}, "latency": {}, "stability": {}}
for (part, fam, tier), g in r0.groupby(["part", "family", "tier"]):
    key = f"{part}|{fam}|{tier}"
    res["groups"][key] = {t: metrics(g[g.target == t]) for t in targets}
    pr = paired(g)
    if pr: res["paired"][key] = pr
for (part, fam), g in r0.groupby(["part", "family"]):   # all tiers pooled
    key = f"{part}|{fam}|all"
    res["groups"][key] = {t: metrics(g[g.target == t]) for t in targets}
    pr = paired(g)
    if pr: res["paired"][key] = pr
cf = r0[r0.part == "cfpb"]
for (fam, sl, tpl), g in cf.groupby(["family", "slice", "template"]):
    res["cfpb_slices"][f"{fam}|{sl}|template={tpl}"] = {t: metrics(g[g.target == t]) for t in targets}
calls = df.drop_duplicates(["target", "rep", "item"])
for (t, part, tier), g in calls[calls.status == 200].groupby(["target", "part", "tier"]):
    res["latency"][f"{t}|{part}|{tier}"] = {"calls": int(len(g)), "p50": float(g.e2e_ms.median()), "p95": float(g.e2e_ms.quantile(.95)), "mean_q": float(g.nq.mean())}
for t, g in df[df.probs.notna()].groupby("target"):
    if g.rep.nunique() > 1:
        res["stability"][t] = float((g.groupby(["item", "qid"]).pred.nunique() == 1).mean())
res["status"] = {t: calls[calls.target == t].status.value_counts().to_dict() for t in targets}

# gate: sysone per-item agreement with published outputs
ref_fn = os.path.join(a.items_dir, "gate_sysone.ref.jsonl")
if os.path.exists(ref_fn):
    ref = [json.loads(l) for l in open(ref_fn, encoding="utf-8") if l.strip()]
    gs = r0[r0.part == "gate_sysone"]; out = {}
    for t, pub in (("jev", "jev"), ("laya-npu", "laya")):
        s = gs[gs.target == t].set_index(["item", "qid"]); agree = correct = n = 0; pub_correct = 0
        for rr in ref:
            k = (rr["id"], rr["qid"])
            if k not in s.index: continue
            row = s.loc[k]; q = items[rr["id"]]["questions"][rr["qid"]]
            keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
            ours = keys[int(row.pred)] if row.pred is not None and not pd.isna(row.pred) else None
            n += 1; agree += str(ours) == str(rr["published"].get(pub)); correct += bool(row.pred == row.gold); pub_correct += bool(rr["published_correct"].get(pub))
        out[t] = {"n": n, "ours_acc": correct / n if n else None, "published_acc": pub_correct / n if n else None, "item_agreement_with_published": agree / n if n else None}
    res["gate_sysone"] = out
s = json.dumps(res, indent=1, default=float)
if a.out: open(a.out, "w", encoding="utf-8").write(s)

# ---- console summary ----
def fmt(m, keys):
    return "  ".join(f"{k}={m[k]:.3f}" if isinstance(m.get(k), float) else f"{k}={m.get(k)}" for k in keys if k in m)
print("status:", res["status"]); print("stability:", res["stability"])
if "gate_sysone" in res: print("gate_sysone:", res["gate_sysone"])
for key, per in sorted(res["groups"].items()):
    part, fam, tier = key.split("|")
    t0 = next(iter(per.values()), {})
    if not any(v.get("n") for v in per.values()): continue
    print(f"\n{key}")
    for t, m in per.items():
        if m.get("n"): print(f"   {t:<9} n={m['n']:<4} " + fmt(m, ["acc", "macro_f1", "balanced_acc", "auroc", "base_rate_yes", "pred_rate_yes", "within_half", "within1", "mae_expected", "qwk", "spearman", "brier", "ece", "soft_brier", "soft_spearman"]))
    if key in res["paired"]: p = res["paired"][key]; print(f"   jev - laya acc diff {p['acc_diff']:+.3f}  CI95 [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  n={p['n']}")
