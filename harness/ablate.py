"""Deterministic item transforms for Laya-fairness ablations of suite-v1.
python3 ablate.py IN.jsonl OUTDIR   -> OUTDIR/suite-v1.{A1..A4}.jsonl (gold remapped so report.py scores them unchanged)
A1 shuffle   : choice options permuted with a seeded RNG keyed on item_id/qid (removes gold-at-index-0 bias). Run Jev AND laya-cpu.
A2 humanize  : choice with null descriptions -> description = key with '_' -> ' ' (no new information); keys untouched.
A3 noul2ch   : noul -> 2-option choice, neutral keys A/B (README Honest-limits workaround); yes/no side counterbalanced by hash.
A4 score2ch  : score -> choice, keys L0..Lk, descriptions = level text (ordinal head removed; same info).
"""
import json, sys, hashlib, random, copy, os
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
items = [json.loads(l) for l in open(src, encoding="utf-8")]
h = lambda *a: int(hashlib.sha256("|".join(map(str, a)).encode()).hexdigest(), 16)

def a1(it):
    for qid, q in it["questions"].items():
        if q["type"] == "choice":
            ks = list(q["criteria"]); random.Random(h(it["id"], qid)).shuffle(ks)
            q["criteria"] = {k: q["criteria"][k] for k in ks}
def a2(it):
    for q in it["questions"].values():
        if q["type"] == "choice":
            q["criteria"] = {k: (v if v else k.replace("_", " ")) for k, v in q["criteria"].items()}
def a3(it):
    for qid, q in it["questions"].items():
        if q["type"] != "noul": continue
        yes_first = h(it["id"], qid) % 2 == 0
        yes, no = "yes, the statement holds", "no, the statement does not hold"
        q["type"] = "choice"; q["criteria"] = {"A": yes, "B": no} if yes_first else {"A": no, "B": yes}
        g = it["gold"].get(qid)
        if g is not None: it["gold"][qid] = ("A" if yes_first else "B") if g is True else ("B" if yes_first else "A")
def a4(it):
    for qid, q in it["questions"].items():
        if q["type"] != "score": continue
        q["type"] = "choice"; q["criteria"] = {"L%d" % i: c for i, c in enumerate(q["criteria"])}
        g = it["gold"].get(qid)
        if g is not None: it["gold"][qid] = "L%d" % int(g)
for name, fn in (("A1", a1), ("A2", a2), ("A3", a3), ("A4", a4)):
    with open(os.path.join(out, "suite-v1.%s.jsonl" % name), "w", encoding="utf-8") as f:
        for it in items:
            it2 = copy.deepcopy(it); fn(it2); f.write(json.dumps(it2, ensure_ascii=False) + "\n")
print("wrote", out)
