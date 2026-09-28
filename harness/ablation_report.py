#!/usr/bin/env python3
"""Accuracy per arm x target x tier x ORIGINAL question type, with input-blind baselines.

python3 tools/ablation_report.py OUT.json ARM=ITEMS.jsonl:RUN_DIR [ARM=...]
  RUN_DIR holds <target>/<target>.jsonl (jevcmp output); rep 0 only.
Baselines (never look at the state): noul -> always true; choice -> first listed option;
score -> the most common gold level overall (2). For ablation arms the same rules apply to the
transformed question (so A1's shuffle removes the first-option advantage).
"""
import json, sys, glob, os, collections

def top(ans, q):
    if not ans: return None
    if q["type"] == "noul": return ans["noul"] >= 0.5
    p = ans.get("probabilities") or {}
    return max(p, key=lambda k: p[k]) if p else None

def gnorm(g, q):
    if q["type"] == "noul": return bool(g)
    if q["type"] == "score": return str(int(g))
    return str(g)

def baseline(q):
    if q["type"] == "noul": return True
    if q["type"] == "choice": return next(iter(q["criteria"]))
    return "2"

out = {}
for spec in sys.argv[2:]:
    arm, rest = spec.split("=", 1); items_path, run = rest.split(":", 1)
    items = {json.loads(l)["id"]: json.loads(l) for l in open(items_path, encoding="utf-8")}
    acc = collections.defaultdict(lambda: [0, 0])
    for fn in glob.glob(os.path.join(run, "*", "*.jsonl")):
        for rec in map(json.loads, open(fn, encoding="utf-8")):
            if rec["rep"] != 0: continue
            it = items[rec["item_id"]]
            for qid, q in it["questions"].items():
                g = (it.get("gold") or {}).get(qid)
                if g is None: continue
                ot = (it["meta"][qid].get("orig_type") or q["type"])
                pred = top((rec.get("answers") or {}).get(qid), q) if rec["status"] == 200 else None
                ok = pred is not None and (str(pred) if q["type"] != "noul" else pred) == gnorm(g, q)
                for key in (f"{rec['target']}|{it['tier']}|{ot}", f"{rec['target']}|{it['tier']}|all"):
                    acc[key][0] += ok; acc[key][1] += 1
    for it in items.values():
        for qid, q in it["questions"].items():
            g = (it.get("gold") or {}).get(qid)
            if g is None: continue
            ot = it["meta"][qid].get("orig_type") or q["type"]; ok = str(baseline(q)) == str(gnorm(g, q)) if q["type"] != "noul" else baseline(q) == gnorm(g, q)
            for key in (f"baseline|{it['tier']}|{ot}", f"baseline|{it['tier']}|all"):
                acc[key][0] += ok; acc[key][1] += 1
    out[arm] = {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(acc.items())}

json.dump(out, open(sys.argv[1], "w"), indent=1)
arms = list(out)
keys = sorted({k for a in out.values() for k in a}, key=lambda k: (k.split("|")[1], k.split("|")[2], k.split("|")[0]))
print(f"{'target|tier|orig_type':<34}" + "".join(f"{a:>14}" for a in arms))
for k in keys:
    print(f"{k:<34}" + "".join(f"{out[a][k]['acc']:>9.3f} n{out[a][k]['n']:<3}" if k in out[a] else f"{'':>14}" for a in arms))
