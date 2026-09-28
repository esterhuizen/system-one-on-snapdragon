#!/usr/bin/env python3
"""prep_items.py OUT.jsonl [--config all] [--split test] [--limit N]  (WSL .venv-analysis)"""
import argparse, json
from datasets import load_dataset
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--config", default="all"); ap.add_argument("--split", default="test")
ap.add_argument("--limit", type=int); a = ap.parse_args()
ds = load_dataset("LocalLLaMA/typed-decisions", a.config, split=a.split); print("columns:", ds.column_names)
n = min(a.limit or len(ds), len(ds))
with open(a.out, "w", encoding="utf-8") as f:
    for i, r in enumerate(ds.select(range(n))):
        gold = None   # TODO: map the label column(s) printed above into {qid: option | level index | bool}
        f.write(json.dumps({"id": f"td-{a.split}-{i}", "suite": f"typed-decisions/{a.config}", "state": json.loads(r["state"]),
                            "questions": json.loads(r["questions"]), "gold": gold}, ensure_ascii=False) + "\n")
