"""Build the FAIR benchmark part `tweets` (plan section 3 B) from the complaint-tweet sources.

python3 build.py            -> writes ../../tweets.jsonl (no tier yet) + sub400.txt + build_stats.json
python3 build.py --tiers M  -> same, plus "tier" per item from the measure JSON M
"""
import os
import argparse, csv, datetime, hashlib, html, json, re, sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

SRC = Path(os.path.expandvars("$WIN_HOME/local-laya/results/fairbench-sources"))
FAIR = Path(os.path.expandvars("$WIN_HOME/local-laya/results/items/fair"))
WORK = FAIR / "_work" / "tweets"
PART = "tweets"

ap = argparse.ArgumentParser(); ap.add_argument("--tiers"); a = ap.parse_args()

tw = list(csv.reader(open(SRC / "complaints-data.csv", encoding="utf-8", newline="")))
sv = list(csv.reader(open(SRC / "complaint_severity.csv", encoding="utf-8", newline="")))
# the samples/ copies are byte-identical (same md5) - assert it
import hashlib as _h
for p in ("samples/complaints-data.csv", "samples/complaint_severity.csv"):
    q = SRC / p; base = SRC / Path(p).name
    assert _h.md5(q.read_bytes()).hexdigest() == _h.md5(base.read_bytes()).hexdigest(), p
assert len(tw) == len(sv) == 3449, (len(tw), len(sv))

def norm(s):
    s = s.lower().replace("<user>", "").replace("<url>", "")
    s = re.sub(r"@\w+", "", s); s = re.sub(r"https?://\S+", "", s)
    s = s.replace("&amp;", "&").replace("& amp ;", "&")
    return re.sub(r"[^a-z0-9]", "", s)

def snowflake_date(tid):
    ms = (int(tid) >> 22) + 1288834974657
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")

COMPLAINT_INS = ("A complaint presents a state of affairs which breaches the writer's favorable expectation. "
                 "Does this message contain a complaint?")
YES = "The message contains a complaint"
NO = "The message does not contain a complaint"
SEV_INS = "How severe is the complaint in this message?"
SEV_LEVELS = [
    "No explicit reproach: states the problem without blaming anyone",
    "Disapproval: expresses dissatisfaction or annoyance",
    "Accusation: says the company did something wrong",
    "Blame: holds the company responsible for the harm",
]
DATASET = ("github.com/danielpreotiuc/complaints-social-media complaints-data.csv (Preotiuc-Pietro et al., ACL 2019) "
           "+ complaint_severity_data.csv (Jin & Aletras, NAACL 2021; archive.org complaint_severity_data), joined by row order")

def yes_to_a(item_id, qid):
    return int(hashlib.sha256((item_id + "|" + qid).encode()).hexdigest(), 16) % 2 == 0

tiers = {}
if a.tiers:
    for r in json.load(open(a.tiers, encoding="utf-8")):
        Ls = [x["L"] for x in r["rows"]]
        # t128: every row <= 128; t256: longest row 129..256 (incl. items whose yes/no rows are <= 128 but
        # whose severity row is 129..256 - no row is truncated, the longest row just needs the 256 bucket);
        # t512: every row 257..512; trunc: any row > 512.
        t = ("trunc" if max(Ls) > 512 else "t128" if max(Ls) <= 128 else
             "t256" if max(Ls) <= 256 else "t512" if min(Ls) > 256 else "mixed")
        tiers[r["id"]] = t
        tiers[r["id"] + "#mixed"] = t == "t256" and min(Ls) <= 128
        assert not r["head_cut"] and not r["no_npu"], r["id"]

items, align, n = [], {"exact": 0, "ratio>=0.95": 0, "low": []}, 0
for line, (t, s) in enumerate(zip(tw, sv)):
    tid, text, lab, dom = t
    if dom in ("random_reply", "random_tweet"):
        continue
    s_id, s_text, s_lab, s_sev, s_dom = s
    # 1:1 alignment checks (row order join): label column, id (severity file has Excel-rounded ids), text
    assert s_lab == lab, (line, lab, s_lab)
    assert lab in ("0", "1") and s_sev in ("0", "1", "2", "3", "4")
    assert (s_sev == "0") == (lab == "0"), (line, lab, s_sev)
    assert abs(float(s_id) - int(tid)) / int(tid) < 0.01, (line, tid, s_id)
    nx, ny = norm(text), norm(s_text)
    if nx == ny: align["exact"] += 1
    else:
        r = SequenceMatcher(None, nx, ny).ratio()
        if r >= 0.95: align["ratio>=0.95"] += 1
        else: align["low"].append({"line": line, "ratio": round(r, 3), "tweet": text, "sev_text": s_text})
    n += 1
    iid = "%s-%04d" % (PART, n)
    state = re.sub(r"\s+", " ", html.unescape(text)).strip()
    is_c = lab == "1"
    a_yes = yes_to_a(iid, "complaint")
    ab_crit = {"A": YES, "B": NO} if a_yes else {"A": NO, "B": YES}
    ab_gold = ("A" if a_yes else "B") if is_c else ("B" if a_yes else "A")
    questions = {
        "complaint": {"type": "noul", "instructions": COMPLAINT_INS, "criteria": {"true": YES, "false": NO}},
        "complaint_ab": {"type": "choice", "instructions": COMPLAINT_INS, "criteria": ab_crit},
        "severity": {"type": "score", "instructions": SEV_INS, "criteria": SEV_LEVELS},
    }
    gold = {"complaint": is_c, "complaint_ab": ab_gold}
    if is_c:
        gold["severity"] = int(s_sev) - 1
    meta = {
        "complaint": {"difficulty": None, "evidence_position": "n/a", "orig_type": "noul", "source_label": lab, "twin_of": None},
        "complaint_ab": {"difficulty": None, "evidence_position": "n/a", "orig_type": "noul", "source_label": lab, "twin_of": "complaint"},
        "severity": {"difficulty": None, "evidence_position": "n/a", "orig_type": "score", "source_label": s_sev, "twin_of": None},
    }
    it = {"id": iid, "suite": PART, "domain": PART, "format": "tweet", "state": state,
          "questions": questions, "gold": gold, "meta": meta,
          "src": {"dataset": DATASET, "row": tid, "date": snowflake_date(tid), "slice": dom}}
    if tiers:
        it["tier"] = tiers[iid]
    items.append(it)

assert n == 1974, n
out = FAIR / "tweets.jsonl"
with open(out, "w", encoding="utf-8", newline="\n") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")

# deterministic 400-item subsample for the yes/no run rule (proportional: complaint rate and severity mix kept)
def h(x): return hashlib.sha256(("sub400|" + x).encode()).hexdigest()
strata = {}
for it in items:
    k = "sev%d" % it["gold"]["severity"] if it["gold"]["complaint"] else "not"
    strata.setdefault(k, []).append(it["id"])
quota = {"not": 150, "sev0": 88, "sev1": 76, "sev2": 46, "sev3": 40}
sub = sorted(i for k, ids in strata.items() for i in sorted(ids, key=h)[: quota[k]])
assert len(sub) == 400
(WORK / "sub400.txt").write_text("\n".join(sub) + "\n", encoding="utf-8")

stats = {
    "items": len(items),
    "complaint_gold": Counter(str(it["gold"]["complaint"]) for it in items),
    "complaint_ab_gold": Counter(it["gold"]["complaint_ab"] for it in items),
    "yes_on_A": sum(it["questions"]["complaint_ab"]["criteria"]["A"] == YES for it in items),
    "severity_gold": Counter(str(it["gold"].get("severity")) for it in items),
    "domain": Counter(it["src"]["slice"] for it in items),
    "date_range": [min(it["src"]["date"] for it in items), max(it["src"]["date"] for it in items)],
    "year": Counter(it["src"]["date"][:4] for it in items),
    "alignment": align,
    "tier": Counter(it.get("tier") for it in items),
    "t256_mixed_ids": [it["id"] for it in items if tiers.get(it["id"] + "#mixed")],
    "t256_all_rows_over_128_ids": [it["id"] for it in items if it.get("tier") == "t256" and not tiers.get(it["id"] + "#mixed")],
    "tier_x_complaint": Counter("%s|%s" % (it.get("tier"), it["gold"]["complaint"]) for it in items),
    "tier_x_severity": Counter("%s|%s" % (it.get("tier"), it["gold"].get("severity")) for it in items),
    "sub400": Counter(k for k, ids in strata.items() for i in ids if i in set(sub)),
    "html_unescaped": sum(it["state"] != re.sub(r"\s+", " ", t[1]).strip() for it, t in
                          zip(items, [t for t in tw if t[3] not in ("random_reply", "random_tweet")])),
}
(WORK / "build_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in stats.items() if k != "alignment"}, ensure_ascii=False))
print("alignment:", align["exact"], align["ratio>=0.95"], len(align["low"]), [(x["line"], x["ratio"], x["sev_text"]) for x in align["low"]])
