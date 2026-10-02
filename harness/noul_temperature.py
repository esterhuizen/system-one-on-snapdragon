"""Yes/no (noul) temperature sweep for decider-12b on the fair suite, from one run with raw letter logits.

python noul_temperature.py RUN_JSONL ITEMS_JSONL   (RUN_JSONL from a jevcmp.py run against npu_decider_serve.py started with DECIDER_NPU_RAW=1)
Each noul answer carries x_raw_logits = soft-capped [no, yes] letter logits; p_yes(T) = sigmoid((yes - no) / T), rounded to
4 decimals as decider.systemone.format_answer does. Reports per task: accuracy, AUROC, Brier, ECE, log loss for each T, and a
leave-one-task-out fitted T (fit on the other tasks' log loss, score on this one).
"""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score

run, items_fn = sys.argv[1], sys.argv[2]
EXCLUDE = {"cfpb-0877", "cfpb-0429", "cfpb-0465"}   # as fair_report.py
items = {it["id"]: it for it in map(json.loads, filter(str.strip, open(items_fn, encoding="utf-8")))}
rows = {}   # task -> list of (delta, gold, served_p)
for rec in map(json.loads, open(run, encoding="utf-8")):
    it = items.get(rec["item_id"])
    if it is None or rec["item_id"] in EXCLUDE or rec["status"] != 200: continue
    for qid, q in it["questions"].items():
        g = (it.get("gold") or {}).get(qid); a = (rec.get("answers") or {}).get(qid)
        if q["type"] != "noul" or g is None or not a or "x_raw_logits" not in a: continue
        no, yes = a["x_raw_logits"]
        rows.setdefault(f"{rec['item_id'].split('-')[0]}|{qid}", []).append((yes - no, int(bool(g)), a["noul"]))

def p_at(d, T): return np.round(1 / (1 + np.exp(-np.clip(d / T, -500, 500))), 4)

def ece(p, y, bins=10):
    conf = np.maximum(p, 1 - p); corr = ((p >= 0.5) == y).astype(float); e = 0.0; edges = np.linspace(0.5, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return e

def stats(d, y, T):
    p = p_at(d, T); pc = np.clip(p, 1e-4, 1 - 1e-4)
    return {"acc": float(((p >= 0.5) == y).mean()), "auroc": float(roc_auc_score(y, p)), "brier": float(np.mean((p - y) ** 2)),
            "ece": float(ece(p, y)), "logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            "saturated": float(np.mean((p < 0.01) | (p > 0.99)))}

TS = [0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
grid = np.exp(np.linspace(np.log(0.05), np.log(20), 200))
out = {}
for task, r in sorted(rows.items()):
    d = np.array([x[0] for x in r]); y = np.array([x[1] for x in r]); served = np.array([x[2] for x in r])
    out[task] = {"n": len(r), "served_auroc": float(roc_auc_score(y, served)), "raw_delta_auroc": float(roc_auc_score(y, d)),
                 "by_T": {str(T): stats(d, y, T) for T in TS}}
# leave-one-task-out temperature (fit on the other tasks' pooled log loss)
for task in (list(out) if len(rows) > 1 else []):
    D = np.concatenate([[x[0] for x in rows[t]] for t in rows if t != task]); Y = np.concatenate([[x[1] for x in rows[t]] for t in rows if t != task])
    Tfit = float(min(grid, key=lambda T: stats(D, Y, T)["logloss"]))
    d = np.array([x[0] for x in rows[task]]); y = np.array([x[1] for x in rows[task]])
    out[task]["loto_T"] = round(Tfit, 3); out[task]["at_loto_T"] = stats(d, y, Tfit)
D = np.concatenate([[x[0] for x in r] for r in rows.values()]); Y = np.concatenate([[x[1] for x in r] for r in rows.values()])
out["_pooled_best_T"] = round(float(min(grid, key=lambda T: stats(D, Y, T)["logloss"])), 3)
print(json.dumps(out, indent=1))
