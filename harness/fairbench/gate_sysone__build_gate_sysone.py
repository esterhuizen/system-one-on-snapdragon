"""Build gate_sysone items + published per-item reference answers.

Source of truth for texts / questions / gold / order: sysone-bench datasets/public_cases.json
(github.com/instax-dutta/sysone-bench @ 1ac3650a65e3783ac9615f26b02bc7c1d5ad3b23).
Independently re-derived here from the primary HF datasets with the repo's own seed-42 sampling
procedure (datasets/build_public.py) and asserted byte-identical, which also yields source row ids.

Published per-item outputs: results/run_jev-1.13.0_20260921-212125.json and
results/run_laya_20260921-213504.json (rows = {qid, ok, conf}, one per case, case order).

Usage: python3 build_gate_sysone.py [--measure MEASURE.json]
  without --measure: writes items with no "tier"; with --measure: adds "tier" from the measure file.
"""
import os
import argparse, hashlib, json, random, sys
from pathlib import Path

W = Path(__file__).resolve().parent
OUTD = W.parent.parent            # results/items/fair
REPO = W / "repo"
SHA = "1ac3650a65e3783ac9615f26b02bc7c1d5ad3b23"
B77_REV = "18072d2685ea682290f7b8924d94c62acc19c0b2"
SST_REV = "e51bdcd8cd3a30da231967c1a249ba59361279a3"
AG_REV = "70e3fa1915be9a8daebec5e840f20df9a8e18793"
JEV_RUN = "results/run_jev-1.13.0_20260921-212125.json"
LAYA_RUN = "results/run_laya_20260921-213504.json"

ap = argparse.ArgumentParser(); ap.add_argument("--measure"); a = ap.parse_args()

pub = json.load(open(REPO / "datasets/public_cases.json", encoding="utf-8"))
assert pub["seed"] == 42
P = pub["suites"]


# ---- 1. independent re-derivation of the sysone public sample (seed 42, build_public.py order) ----
def rederive():
    sys.path.insert(0, os.path.expandvars("$WIN_HOME/local-laya/results/fairbench-sources/pylib_ns"))
    import pyarrow.parquet as pq
    rng = random.Random(42)
    ag = [json.loads(l) for l in open(W / "agnews_test.jsonl", encoding="utf-8")]
    lab = ["world", "sports", "business", "scitech"]
    idx = rng.sample(range(len(ag)), 100)
    assert [[{"text": ag[i]["title"] + " - " + ag[i]["description"]}, {"topic": lab[ag[i]["label"] - 1]}] for i in idx] == P["agnews"]["cases"]
    em = pq.read_table(os.path.expandvars("$WIN_HOME/local-laya/results/fairbench-sources/ns_data/dair_test.parquet")).to_pylist()
    el = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    idx = rng.sample(range(len(em)), 100)
    assert [[{"text": em[i]["text"]}, {"emotion": el[em[i]["label"]]}] for i in idx] == P["emotion"]["cases"]
    intents = list(P["banking77_12"]["questions"]["intent"]["criteria"])
    bk = pq.read_table(W / "b77_test.parquet").to_pylist()          # = load_dataset("mteb/banking77", split="test"), 3076 rows
    assert len(bk) == 3076
    by = {}
    for ri, r in enumerate(bk):
        if r["label_text"].lower() in intents:
            by.setdefault(r["label_text"].lower(), []).append((ri, r))
    cases = []
    for it in intents:
        for ri, r in rng.sample(by[it], 8):
            cases.append((ri, r, it))
    rng.shuffle(cases)
    assert [[{"text": r["text"]}, {"intent": it}] for ri, r, it in cases] == P["banking77_12"]["cases"]
    rng.sample(range(9815), 80)                                    # mnli validation_matched draw (len 9815)
    sst = [json.loads(l) for l in open(W / "sst5_test.jsonl", encoding="utf-8")]
    assert len(sst) == 2210
    idx = rng.sample(range(len(sst)), 60)
    assert [[{"text": sst[i]["text"]}, {"sentiment": int(sst[i]["label"])}] for i in idx] == P["sst5"]["cases"]
    jl = [json.loads(l) for l in open(W / "b77_test.jsonl", encoding="utf-8")]   # test.jsonl (3080 rows), for cross-ref
    jidx = {}
    for i, r in enumerate(jl):
        jidx.setdefault((r["text"], r["label_text"]), []).append(i)
    b_src = [{"row": ri, "row_testjsonl": jidx[(r["text"], r["label_text"])], "label": r["label"], "label_text": r["label_text"]}
             for ri, r, it in cases]
    s_src = [{"row": i, "label": sst[i]["label"], "label_text": sst[i]["label_text"]} for i in idx]
    return b_src, s_src


b_src, s_src = rederive()


def qhash(q):   # sysone run.py question_hash
    return hashlib.sha256(json.dumps(q, sort_keys=True).encode()).hexdigest()[:12]


J = json.load(open(REPO / JEV_RUN, encoding="utf-8"))
L = json.load(open(REPO / LAYA_RUN, encoding="utf-8"))
HASH = {"banking77_12": "71f28b510c31", "sst5": "621c66266036"}
for s in HASH:
    assert qhash(P[s]["questions"]) == HASH[s] == J["meta"]["question_hash"][s] == L["meta"]["question_hash"][s]

tiers = {}
if a.measure:
    for r in json.load(open(a.measure, encoding="utf-8")):
        Ls = [x["L"] for x in r["rows"]]
        assert not r["head_cut"] and not r["no_npu"], r["id"]
        tiers[r["id"]] = ("trunc" if max(Ls) > 512 else "t128" if max(Ls) <= 128 else
                          "t256" if min(Ls) > 128 and max(Ls) <= 256 else
                          "t512" if min(Ls) > 256 and max(Ls) <= 512 else "mixed")

items, refs, n = [], [], 0
for s, suite, fmt, src_list, ds in [
        ("banking77_12", "gate_sysone/banking77", "message", b_src,
         f"mteb/banking77@{B77_REV} data/test-00000-of-00001.parquet (datasets.load_dataset default config, test split, 3076 rows)"),
        ("sst5", "gate_sysone/sst5", "post", s_src,
         f"SetFit/sst5@{SST_REV} test.jsonl (datasets.load_dataset, test split, 2210 rows)")]:
    spec = P[s]; (qid, q), = spec["questions"].items()
    jr, lr = J["suites"][s]["rows"], L["suites"][s]["rows"]
    assert len(jr) == len(lr) == len(spec["cases"]) == len(src_list)
    for ci, ((state, exp), src, j, l) in enumerate(zip(spec["cases"], src_list, jr, lr)):
        n += 1; iid = f"gate_sysone-{n:04d}"
        assert j["qid"] == l["qid"] == qid and list(exp) == [qid]
        gold = exp[qid]
        item = {"id": iid, "suite": suite, "domain": "gate_sysone", "format": fmt,
                "state": state,                                   # sysone state dict, byte-identical ({"text": ...})
                "questions": {qid: q},                            # sysone question dict, byte-identical (same key order)
                "gold": {qid: gold},
                "meta": {qid: {"difficulty": None, "evidence_position": "n/a", "orig_type": q["type"],
                               "source_label": src["label_text"], "twin_of": None}},
                "src": {"dataset": ds, "row": str(src["row"]), "date": None,
                        "slice": f"sysone-bench@{SHA[:7]} datasets/public_cases.json suites.{s}.cases[{ci}] (seed 42)",
                        "sysone_suite": s, "sysone_case": ci, "sysone_question_hash": HASH[s]}}
        if "row_testjsonl" in src:
            item["src"]["row_testjsonl"] = ",".join(map(str, src["row_testjsonl"]))
        if a.measure:
            item["tier"] = tiers[iid]
        items.append(item)
        refs.append({"id": iid, "qid": qid, "gold": gold,
                     "published": {"jev": gold if j["ok"] else None, "laya": gold if l["ok"] else None},
                     "published_correct": {"jev": bool(j["ok"]), "laya": bool(l["ok"])},
                     "published_conf": {"jev": j["conf"], "laya": l["conf"]},
                     "suite": suite, "sysone_case": ci})

# sanity: question hash of each item's own questions dict matches the published hash
for it in items:
    assert qhash(it["questions"]) == it["src"]["sysone_question_hash"]

with open(OUTD / "gate_sysone.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
with open(OUTD / "gate_sysone.ref.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in refs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# aggregate check vs published
for suite, s in [("gate_sysone/banking77", "banking77_12"), ("gate_sysone/sst5", "sst5")]:
    rr = [r for r in refs if r["suite"] == suite]
    for m, run in [("jev", J), ("laya", L)]:
        acc = sum(r["published_correct"][m] for r in rr) / len(rr)
        assert round(acc, 4) == run["suites"][s]["accuracy"], (suite, m, acc)
        print(f"{suite:<24} {m:<5} n={len(rr)} acc={acc:.4f} (published {run['suites'][s]['accuracy']})")
print("items", len(items), "refs", len(refs), "tiers", {t: sum(1 for i in items if i.get('tier') == t) for t in set(tiers.values())})
