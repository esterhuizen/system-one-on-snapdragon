"""Fetch SetFit/enron_spam (all splits) via HF datasets-server /rows, 100 rows per page.
Writes enron_spam_all.jsonl with split, row_idx, message_id, label_text, text, and a truncation flag."""
import json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enron_spam_all.jsonl")
BASE = "https://datasets-server.huggingface.co/rows?dataset=SetFit/enron_spam&config=default&split={split}&offset={off}&length=100"
SPLITS = {"train": 31716, "test": 2000}


def get(split, off):
    url = BASE.format(split=split, off=off)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                d = json.load(r)
            return split, off, d["rows"]
        except Exception as e:  # 429 / 5xx: back off
            time.sleep(2 * (attempt + 1))
            err = e
    raise RuntimeError(f"failed {split} {off}: {err}")


jobs = [(s, o) for s, n in SPLITS.items() for o in range(0, n, 100)]
res = {}
with ThreadPoolExecutor(6) as ex:
    for split, off, rows in ex.map(lambda a: get(*a), jobs):
        res[(split, off)] = rows
n = 0
trunc = 0
with open(OUT, "w", encoding="utf-8") as f:
    for split, off in jobs:
        for r in res[(split, off)]:
            row = r["row"]
            tc = r.get("truncated_cells") or []
            trunc += bool(tc)
            f.write(json.dumps({"split": split, "row_idx": r["row_idx"], "message_id": row["message_id"],
                                "label_text": row["label_text"], "subject": row["subject"],
                                "message": row["message"], "text": row["text"], "truncated_cells": tc},
                               ensure_ascii=False) + "\n")
            n += 1
print("rows", n, "truncated", trunc, {s: sum(1 for (sp, o) in jobs if sp == s) for s in SPLITS})
