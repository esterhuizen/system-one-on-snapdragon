"""Re-score JevBench yes/no (noul) answers with the frozen v1.5 rule (docs/METHOD-v1.5.md §3.1): P(yes) <= 0.20 is No,
>= 0.80 is Yes, anything in between is an abstention counted wrong; competence CC = 100 * (acc - 0.5) / 0.5.
Compared with the open-source scorer (argmax). Tier weights easy 0.10, standard 0.20, hard 0.40 (no judge tier in the
public set, renormalised). For the NPU 12B runs only answered items count; the "common" rows use the items every system
answered.      JEVBENCH_RUNS=... JEVBENCH_PUBLIC=... python3 jevbench_v15_noul.py
"""
import json, collections
import os
R = os.environ.get("JEVBENCH_RUNS", "runs"); D = os.environ.get("JEVBENCH_PUBLIC", "jevbench/datasets/public")   # run dirs as written by run_jevbench; the public items
SYS = {"Jev 1.13.0": "full/jev", "decider-2b (CPU)": "full/decider", "imajev-2b (CPU)": "full/imajev", "Laya (NPU)": "full/laya-npu",
       "Winnow-12B (NPU)": "pub2/winnow-npu", "decider-12b (NPU)": "pub/decider-npu"}
TIER = {"easy": "easy", "original": "standard", "hard": "hard"}; WT = {"easy": 0.10, "standard": 0.20, "hard": 0.40}
gold = {}
for fn in ("easy", "original", "hard"):
    for l in open(f"{D}/{fn}.jsonl"):
        t = json.loads(l)
        if t["question"]["type"] == "noul": gold[t["id"] if "id" in t else t["task_id"]] = (TIER[fn], t["expected"])
res = {}
for name, d in SYS.items():
    res[name] = {}
    for l in open(f"{R}/{d}/results.jsonl"):
        x = json.loads(l); tid = x["task_id"]
        if tid not in gold or x["status"] != "ok" or not x.get("probs"): continue
        p = float(x["probs"].get("yes", 0.0)); exp = gold[tid][1]
        arg = ("yes" if p >= 0.5 else "no") == exp
        band = ("yes" if p >= 0.8 else "no" if p <= 0.2 else None) == exp
        res[name][tid] = (arg, band, 0.2 < p < 0.8)
common = set.intersection(*(set(v) for v in res.values()))
def table(ids_of, title):
    print(f"\n{title}\n| System | yes/no items | argmax accuracy | v1.5 accuracy | answers in 0.2-0.8 | v1.5 Noul CC | argmax CC |")
    print("|---|---|---|---|---|---|---|")
    for name, r in res.items():
        ids = ids_of(r); by = collections.defaultdict(list)
        for i in ids: by[gold[i][0]].append(r[i])
        acc = lambda k, rows: sum(z[k] for z in rows) / len(rows)
        w = sum(WT[t] for t in by)
        cc = lambda k: sum(WT[t] * 100 * (acc(k, rows) - 0.5) / 0.5 for t, rows in by.items()) / w
        allr = [r[i] for i in ids]
        print(f"| {name} | {len(allr)} | {acc(0, allr):.3f} | {acc(1, allr):.3f} | {sum(z[2] for z in allr) / len(allr):.1%} | {cc(1):.1f} | {cc(0):.1f} |")
table(lambda r: set(r), "All yes/no items each system answered (74 in the public set)")
table(lambda r: common & set(r), f"Common yes/no items answered by every system ({len(common)})")
