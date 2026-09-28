"""Build results/items/fair/enron_ffp.jsonl (plan section 3 D, ENRON-FFP).

python3 build.py                       -> writes ../../enron_ffp.jsonl without tiers
python3 build.py --tiers MEASURE.json  -> same items, plus "tier" from the Laya length tool output
Deterministic. Requires overlap.json (from overlap.py) in this directory.
"""
import argparse, csv, hashlib, json, os, re, statistics, collections

HERE = os.path.dirname(os.path.abspath(__file__))
FAIR = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(FAIR, "enron_ffp.jsonl")
FFP = os.path.expandvars("$WIN_HOME/local-laya/results/fairbench-sources/ns_data/enron_ffp/enron-FFP.csv")
PART = "enron_ffp"
DATASET = ("ENRON-FFP enron-FFP.csv (github.com/kushalchawla/Frustration-Prediction-In-Emails -> "
           "bit.ly/2IAxPab Dropbox zip; Chhaya et al. 2018 PEOPLES@NAACL / Khosla et al. COLING 2018), "
           "minus emails overlapping SetFit/enron_spam@1916f66c (normalised 50-char window match)")

Q_FRUS = "Does the writer of this email sound frustrated or annoyed?"
YES_FRUS, NO_FRUS = "Frustrated, annoyed, or impatient", "Calm, neutral, or positive"
Q_POL = "How polite is this email?"
POL_LEVELS = ["Impolite", "Neutral", "Polite"]


def clean(s):
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(l.rstrip() for l in s.split("\n")).strip()
    return re.sub(r"\n{3,}", "\n\n", s)


def ratings(row, pre):
    raw = [row[f"{pre}{i}"].strip() for i in range(1, 11)]
    vals = [float(x) for x in raw if x != ""]
    lab = ",".join("NA" if x == "" else str(int(float(x))) for x in raw)
    return vals, lab


def yes_on_a(item_id, qid):
    return int(hashlib.sha256((item_id + "|" + qid).encode()).hexdigest(), 16) % 2 == 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tiers")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="drop an email when its share of 50-char windows found in enron_spam is > 0 (default: any "
                         "shared window) or >= this value if > 0 (e.g. 0.3 keeps emails sharing only signatures/boilerplate)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    tiers = None
    if a.tiers:
        tiers = {}
        for rec in json.load(open(a.tiers, encoding="utf-8")):
            Ls = [r["L"] for r in rec["rows"]]
            if max(Ls) > 512: t = "trunc"
            elif max(Ls) <= 128: t = "t128"
            elif max(Ls) <= 256: t = "t256"      # all rows <=256, at least one 129..256 (longest-row rule)
            else: t = "t512"                     # longest row 257..512
            tiers[rec["id"]] = t

    rows = list(csv.DictReader(open(FFP, encoding="utf-8", newline="")))
    ov = {o["row"]: o for o in json.load(open(os.path.join(HERE, "overlap.json")))}
    items, stats = [], collections.Counter()
    n = 0
    for ri, row in enumerate(rows):
        if (ov[ri]["overlap"] if a.min_coverage <= 0 else ov[ri]["coverage"] >= a.min_coverage):
            stats["dropped_spam_overlap"] += 1
            continue
        n += 1
        iid = f"{PART}-{n:04d}"
        fv, flab = ratings(row, "Frustration")
        pv, plab = ratings(row, "Politeness")
        share = round(sum(1 for x in fv if x < 0) / len(fv), 4)
        pmean = round(statistics.mean(pv), 4)
        gold_frus = True if share >= 0.3 else (False if share == 0 else None)
        pol_gold = 0 if pmean <= -0.5 else (2 if pmean >= 0.5 else 1)

        a_yes = yes_on_a(iid, "frustrated")
        ab_crit = {"A": YES_FRUS, "B": NO_FRUS} if a_yes else {"A": NO_FRUS, "B": YES_FRUS}
        questions = {
            "frustrated": {"type": "noul", "instructions": Q_FRUS, "criteria": {"true": YES_FRUS, "false": NO_FRUS}},
            "frustrated_ab": {"type": "choice", "instructions": Q_FRUS, "criteria": ab_crit},
            "politeness": {"type": "score", "instructions": Q_POL, "criteria": list(POL_LEVELS)},
        }
        gold = {}
        if gold_frus is not None:
            gold["frustrated"] = gold_frus
            gold["frustrated_ab"] = ("A" if a_yes else "B") if gold_frus else ("B" if a_yes else "A")
        gold["politeness"] = pol_gold
        m = lambda ot, lab, sg, tw: {"difficulty": None, "evidence_position": "n/a", "orig_type": ot,
                                    "source_label": lab, "soft_gold": sg, "twin_of": tw}
        meta = {
            "frustrated": m("noul", flab, share, None),
            "frustrated_ab": m("noul", flab, share, "frustrated"),
            "politeness": m("score", plab, pmean, None),
        }
        it = {"id": iid, "suite": PART, "domain": PART, "format": "email", "state": clean(row["Email"]),
              "questions": questions, "gold": gold, "meta": meta,
              "src": {"dataset": DATASET, "row": str(ri), "date": "", "slice": ""}}
        if tiers is not None:
            it["tier"] = tiers[iid]
        items.append(it)
        stats["kept"] += 1
        stats[f"frustrated={gold_frus}"] += 1
        stats[f"politeness={POL_LEVELS[pol_gold]}"] += 1
        stats[f"yes_on_{'A' if a_yes else 'B'}"] += 1
        if "frustrated_ab" in gold: stats[f"frustrated_ab={gold['frustrated_ab']}"] += 1
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(dict(sorted(stats.items())))


if __name__ == "__main__":
    main()
