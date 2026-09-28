#!/usr/bin/env python3
"""Build gate_td.jsonl: LocalLLaMA/typed-decisions TEST split (config=all) -> harness item format.

Verbatim: state (JSON string decoded to the object the dataset card's usage example gives),
question instructions/criteria (decoded, criteria key order preserved = option order), gold =
dataset's discrete teacher label mapped choice->key, noul->bool, score->int level index.
No yes/no twins (harness gate: reproduce the published number).

usage: python3 build_gate_td.py [--tiers MEASURE.json]   (second pass adds "tier")
"""
import os
import argparse, json, sys
from pathlib import Path

SRC = Path(os.path.expandvars("$WIN_HOME/local-laya/results/fairbench-sources/vendor/td/test_all.json"))
OUT = Path(os.path.expandvars("$WIN_HOME/local-laya/results/items/fair/gate_td.jsonl"))
REV = "f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8"          # HF main at build time
DATA_COMMIT = "468b1461d59e01d6404e1cf37431da962acd43a0"   # last commit touching all/test parquet
PARQUET_SHA256 = "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"
PART = "gate_td"

ap = argparse.ArgumentParser(); ap.add_argument("--tiers"); a = ap.parse_args()


def tier_of(rows_L):
    mx = max(rows_L)
    if mx > 512: return "trunc"
    if mx <= 128: return "t128"
    if min(rows_L) > 256: return "t512"
    if min(rows_L) > 128 and mx <= 256: return "t256"
    return None   # mixed across a bucket boundary; resolved below


measured = None
if a.tiers:
    measured = {r["id"]: r for r in json.loads(Path(a.tiers).read_text(encoding="utf-8"))}

rows = json.loads(SRC.read_text(encoding="utf-8"))
assert len(rows) == 400
items = []
for i, r in enumerate(rows):
    assert r["split"] == "test"
    state = json.loads(r["state"])
    qs = json.loads(r["questions"])
    g = json.loads(r["gold"])
    la = json.loads(r["label_agreement"])
    iid = f"{PART}-{i:04d}"
    questions, gold, meta = {}, {}, {}
    for qid, qd in qs.items():
        t = qd["type"]
        q = {"type": t, "instructions": qd["instructions"]}
        if "criteria" in qd:
            q["criteria"] = qd["criteria"]
        extra = set(qd) - {"type", "instructions", "criteria"}
        assert not extra, (r["id"], qid, extra)
        questions[qid] = q
        gg = g[qid]; assert gg["type"] == t
        lab, probs = gg["label"], gg["probabilities"]
        if t == "choice":
            assert lab in qd["criteria"]; gold[qid] = lab; soft = probs[lab]
        elif t == "noul":
            assert lab in ("true", "false"); gold[qid] = (lab == "true"); soft = gg["noul"]
        elif t == "score":
            k = int(lab); assert 0 <= k < len(qd["criteria"]) and str(k) == lab; gold[qid] = k; soft = probs[lab]
        else:
            raise SystemExit(f"unknown type {t}")
        mx = max(probs.values())
        m = {"difficulty": None, "evidence_position": "n/a", "orig_type": t, "source_label": lab,
             "soft_gold": soft, "twin_of": None,
             "gold_probabilities": probs, "gold_confidence": gg["confidence"],
             "gold_tie": sum(abs(v - mx) < 1e-9 for v in probs.values()) > 1,
             "teacher_argmax_agree": la[qid]["argmax_agree"],
             "teacher_total_variation": la[qid]["total_variation"]}
        if t == "score":
            m["expected_score"] = gg["score"]
        meta[qid] = m
    it = {"id": iid, "suite": f"{PART}/{r['workflow']}", "domain": PART, "format": "td",
          "state": state, "questions": questions, "gold": gold, "meta": meta,
          "src": {"dataset": "LocalLLaMA/typed-decisions", "config": "all", "split": "test",
                  "revision": REV, "data_commit": DATA_COMMIT, "parquet_sha256": PARQUET_SHA256,
                  "row": r["id"], "row_idx": i, "date": None, "slice": r["workflow"]}}
    if measured is not None:
        mr = measured[iid]
        Ls = [x["L"] for x in mr["rows"]]
        assert [x["qid"] for x in mr["rows"]] == list(questions)
        tier = tier_of(Ls)
        if tier is None:
            # straddles a bucket boundary (not covered by the four tier names): tier by the longest row,
            # which is the bucket the whole state must fit in; flagged so it can be split out.
            mx = max(Ls); tier = "t256" if mx <= 256 else "t512"
            it["tier_mixed"] = True
        it["tier"] = tier
        # measured with scripts/measure_items.py (exact Laya/NPU prompt builder); plan §2 asks for
        # results with and without the states longer than ~320 Laya tokens (512 - 192 head budget)
        it["laya_len"] = {"state_tokens": mr["state_tokens"], "L_min": min(Ls), "L_max": max(Ls),
                          "rows_over_512": [x["qid"] for x in mr["rows"] if x["L"] > 512],
                          "state_gt320": mr["state_tokens"] > 320}
    items.append(it)

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(f"wrote {len(items)} items -> {OUT}", file=sys.stderr)
