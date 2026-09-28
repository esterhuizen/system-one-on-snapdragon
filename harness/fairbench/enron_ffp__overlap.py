"""Find ENRON-FFP emails overlapping SetFit/enron_spam (train+test, ham+spam).
Normalisation: lowercase, keep only [a-z0-9] (drops whitespace, punctuation, non-ASCII) -- enron_spam text is
lowercased and space-tokenised ("don ' t", "www . x . com"), so whitespace/punctuation must be ignored.
Match: an FFP email overlaps if ANY 50-character window of its normalised text occurs in any normalised
enron_spam `text` (subject + message). Emails whose normalised text is shorter than 50 chars are matched
as a whole-string substring.
Output: overlap.json with, per FFP row: norm_len, n_windows, matched_windows, coverage, spam ids."""
import csv, json, os, re, collections

HERE = os.path.dirname(os.path.abspath(__file__))
FFP = os.path.expandvars("$WIN_HOME/local-laya/results/fairbench-sources/ns_data/enron_ffp/enron-FFP.csv")
K = 50
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())

rows = list(csv.DictReader(open(FFP, encoding="utf-8", newline="")))
ffp = [norm(r["Email"]) for r in rows]
win = collections.defaultdict(list)  # window -> [(ffp_idx, pos)]
short = []
for i, n in enumerate(ffp):
    if len(n) < K:
        short.append(i)
        continue
    for p in range(len(n) - K + 1):
        win[n[p:p + K]].append((i, p))

matched = collections.defaultdict(set)   # ffp_idx -> positions matched
spam_ids = collections.defaultdict(set)  # ffp_idx -> "split:message_id:label"
corpus = []
for split in ("train", "test"):
    for line in open(os.path.join(HERE, f"enron_spam_{split}.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        t = norm(d["text"] or "")
        corpus.append((f"{split}:{d['message_id']}:{d['label_text']}", t))
        seen = set()
        for p in range(len(t) - K + 1):
            hit = win.get(t[p:p + K])
            if hit:
                for i, q in hit:
                    matched[i].add(q)
                    seen.add(i)
        for i in seen:
            spam_ids[i].add(corpus[-1][0])
for i in short:
    if not ffp[i]:
        continue
    for sid, t in corpus:
        if ffp[i] in t:
            matched[i].add(0)
            spam_ids[i].add(sid)

out = []
for i, n in enumerate(ffp):
    nw = max(len(n) - K + 1, 1)
    mw = len(matched.get(i, ()))
    out.append({"row": i, "norm_len": len(n), "n_windows": nw, "matched_windows": mw,
                "coverage": round(mw / nw, 4), "overlap": mw > 0,
                "spam_ids": sorted(spam_ids.get(i, ()))[:20], "n_spam_ids": len(spam_ids.get(i, ()))})
json.dump(out, open(os.path.join(HERE, "overlap.json"), "w"), indent=0)
ov = [o for o in out if o["overlap"]]
print("ffp", len(out), "short(<50 norm chars)", len(short), "overlap", len(ov), "remain", len(out) - len(ov))
import statistics
cov = sorted(o["coverage"] for o in ov)
print("coverage quantiles among overlapping:", [cov[int(q * (len(cov) - 1))] for q in (0, .1, .25, .5, .75, .9, 1)] if cov else None)
print("coverage < 0.5:", sum(c < 0.5 for c in cov), " < 0.2:", sum(c < 0.2 for c in cov))
