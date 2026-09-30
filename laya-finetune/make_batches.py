"""Split items into labelling batches and a held-out evaluation set.

items.jsonl: one item per line, {"id": "...", <field>: <value>, ...}  (only the fields the model should read)
python make_batches.py --items items.jsonl --out-dir WORK [--batch 100] [--holdout 200] [--seed 7]
Writes WORK/batches/<name>.jsonl (t00, t01, ... training; v00, ... held-out), WORK/train_ids.txt, WORK/holdout_ids.txt,
and WORK/batches/index.json. Held-out items are labelled too (to measure label quality) but never used for training.
"""
import argparse, json, os, random

ap = argparse.ArgumentParser(); ap.add_argument("--items", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--batch", type=int, default=100); ap.add_argument("--holdout", type=int, default=200); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
items = [json.loads(l) for l in open(a.items, encoding="utf-8") if l.strip()]
ids = [it["id"] for it in items]; by = {it["id"]: it for it in items}
rng = random.Random(a.seed); shuffled = ids[:]; rng.shuffle(shuffled)
hold = set(shuffled[: a.holdout]); train = [i for i in ids if i not in hold]; held = [i for i in ids if i in hold]
os.makedirs(os.path.join(a.out_dir, "batches"), exist_ok=True)
index = []
for prefix, group in (("t", train), ("v", held)):
    for k in range(0, len(group), a.batch):
        name = f"{prefix}{k // a.batch:02d}"; index.append({"batch": name, "n": len(group[k: k + a.batch])})
        with open(os.path.join(a.out_dir, "batches", name + ".jsonl"), "w", encoding="utf-8") as f:
            for i in group[k: k + a.batch]: f.write(json.dumps(by[i], ensure_ascii=False) + "\n")
open(os.path.join(a.out_dir, "train_ids.txt"), "w").write("\n".join(train) + "\n")
open(os.path.join(a.out_dir, "holdout_ids.txt"), "w").write("\n".join(held) + "\n")
json.dump(index, open(os.path.join(a.out_dir, "batches", "index.json"), "w"), indent=1)
print(f"{len(train)} training + {len(held)} held-out items in {len(index)} batches -> {a.out_dir}; batch names: {[b['batch'] for b in index]}")
