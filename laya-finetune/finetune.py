"""Fine-tune Laya's encoder (ModernBERT-large, as trained by Convai Innovations) plus one linear head per task on your labels,
on the CPU. Heads start from multinomial logistic regression fitted on the frozen, mean-pooled encoder output, so training
refines an already-good classifier (the first line printed, "epoch 0", is that quick test on its own).

python finetune.py --items items.jsonl --labels-dir WORK/final --taxonomy taxonomy.json --train-ids WORK/train_ids.txt
                   --out-dir OUT [--holdout-ids WORK/holdout_ids.txt] [--holdout-gold gold.jsonl]
                   [--text-template "{summary}\\nType: {request_type}\\nCategory: {helpdesk_category}"]
                   [--epochs 3] [--batch 16] [--lr 2e-5] [--head-lr 5e-4] [--max-len 64] [--threads 12] [--max-train N]

- items.jsonl      {"id", <fields>...}; text = --text-template filled from the fields (missing fields -> "")
- labels           WORK/final/*.jsonl lines {"id", <task>: <key>, "confidence": high|medium|low} (label_workflow.js)
- taxonomy.json    {"tasks": {"<task>": {"<key>": "<definition>", ...}, ...}}
- training ids     minus a 10% dev split (seeded) that picks the best epoch; held-out ids are scored at the end only,
                   against --holdout-gold (human labels, JSONL {"id", <task>: <key>}) if given, else against the labels
Low-confidence labels count less (high 1.0, medium 0.8, low 0.5). Saves OUT/classifier.safetensors, classifier_config.json,
report.json.
"""
import argparse, glob, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import laya
from safetensors.torch import save_file
from transformers import AutoTokenizer

try:
    import nothrottle  # noqa: F401  (Windows: opt out of EcoQoS so a background run is not throttled)
except ImportError:
    pass

LAYA_REPO, LAYA_REV = "convaiinnovations/laya", "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
ap = argparse.ArgumentParser()
for k in ("items", "labels-dir", "taxonomy", "train-ids", "out-dir"): ap.add_argument("--" + k, required=True)
ap.add_argument("--holdout-ids"); ap.add_argument("--holdout-gold")
ap.add_argument("--text-template", default="{summary}\nType: {request_type}\nCategory: {helpdesk_category}")
ap.add_argument("--snapshot", help="local Laya snapshot dir (default: download convaiinnovations/laya at the pinned revision)")
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--head-lr", type=float, default=5e-4)
ap.add_argument("--max-len", type=int, default=64); ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--seed", type=int, default=13); ap.add_argument("--max-train", type=int, help="use only N training items (smoke test)")
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); random.seed(a.seed)
snap = a.snapshot
if not snap:
    from huggingface_hub import snapshot_download
    snap = snapshot_download(LAYA_REPO, revision=LAYA_REV)

TASKS = {t: list(keys) for t, keys in json.load(open(a.taxonomy, encoding="utf-8"))["tasks"].items()}
items = {it["id"]: it for it in map(json.loads, open(a.items, encoding="utf-8"))}


class _Blank(dict):
    def __missing__(self, k): return ""


text = lambda it: a.text_template.format_map(_Blank({k: ("" if v is None else v) for k, v in it.items()}))
lab = {}
for f in glob.glob(os.path.join(a.labels_dir, "*.jsonl")):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l); lab[r["id"]] = r
hold = open(a.holdout_ids).read().split() if a.holdout_ids else []
gold = {r["id"]: r for r in map(json.loads, open(a.holdout_gold, encoding="utf-8"))} if a.holdout_gold else {i: lab[i] for i in hold if i in lab}
train = [i for i in open(a.train_ids).read().split() if i in lab and i in items and i not in set(hold)]
random.Random(a.seed).shuffle(train); ndev = max(1, len(train) // 10); dev, train = train[:ndev], train[ndev:]
if a.max_train: train = train[: a.max_train]; dev = dev[: max(20, a.max_train // 10)]
hold = [i for i in hold if i in gold and i in items]
W = {"high": 1.0, "medium": 0.8, "low": 0.5}
tok = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))


class Classifier(nn.Module):
    def __init__(self, enc):
        super().__init__(); self.enc = enc; d = enc.config.hidden_size; self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({t: nn.Linear(d, len(k)) for t, k in TASKS.items()})

    def pooled(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state; m = mask[..., None].float()
        return (h * m).sum(1) / m.sum(1)

    def forward(self, ids, mask):
        p = self.drop(self.pooled(ids, mask)); return {t: h(p) for t, h in self.heads.items()}


model = Classifier(laya.load(snap).model.encoder)


def batches(keys, bs, shuffle):
    keys = keys[:]
    if shuffle: random.shuffle(keys)
    for i in range(0, len(keys), bs):
        k = keys[i: i + bs]; b = tok([text(items[x]) for x in k], padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
        yield k, b["input_ids"], b["attention_mask"]


@torch.inference_mode()
def predict(keys):
    model.eval(); out = {}
    for k, ids, mask in batches(keys, 64, False):
        lg = model(ids, mask)
        for t in TASKS:
            p = lg[t].softmax(-1)
            for j, x in enumerate(k): out.setdefault(x, {})[t] = (TASKS[t][int(p[j].argmax())], float(p[j].max()))
    return out


def score(pred, keys, ref, tag):
    r = {}
    for t in TASKS:
        ks = [k for k in keys if ref[k].get(t)]
        if ks:
            r[f"{tag}_{t}"] = round(float(np.mean([pred[k][t][0] == ref[k][t] for k in ks])), 4)
            hi = [k for k in ks if pred[k][t][1] >= 0.7]
            r[f"{tag}_{t}_share_p70"] = round(len(hi) / len(ks), 3)
            r[f"{tag}_{t}_acc_p70"] = round(float(np.mean([pred[k][t][0] == ref[k][t] for k in hi])), 4) if hi else None
    return r


t0 = time.perf_counter(); model.eval(); feats = []
with torch.inference_mode():
    for k, ids, mask in batches(train, 64, False): feats.append(model.pooled(ids, mask))
X = torch.cat(feats); mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd
for t, head in model.heads.items():      # multinomial logistic regression (L-BFGS, L2), folded into the head
    y = torch.tensor([TASKS[t].index(lab[k][t]) for k in train]); lin = nn.Linear(Z.shape[1], len(TASKS[t]))
    lb = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=300, line_search_fn="strong_wolfe")
    def closure():
        lb.zero_grad(); loss = nn.functional.cross_entropy(lin(Z), y) + 1e-2 * lin.weight.pow(2).sum(); loss.backward(); return loss
    lb.step(closure)
    with torch.no_grad():
        head.weight.copy_(lin.weight / sd); head.bias.copy_(lin.bias - (lin.weight * (mu / sd)).sum(1))
report = {"n_train": len(train), "n_dev": len(dev), "n_holdout": len(hold), "holdout_ref": "gold" if a.holdout_gold else "labels",
          "init_seconds": round(time.perf_counter() - t0), "args": vars(a)}
p0 = predict(dev + hold); report["epoch0"] = {**score(p0, dev, lab, "dev"), **score(p0, hold, gold, "holdout")}
print("epoch 0 (heads only, i.e. the quick test):", report["epoch0"], flush=True)

opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": a.lr}, {"params": model.heads.parameters(), "lr": a.head_lr}], weight_decay=0.01)
steps = a.epochs * math.ceil(len(train) / a.batch); warm = int(0.06 * steps)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (steps - s) / max(1, steps - warm)))
ce = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.05)


def save():
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(a.out_dir, "classifier.safetensors"))
    json.dump({"tasks": TASKS, "max_len": a.max_len, "text_template": a.text_template, "laya_repo": LAYA_REPO, "laya_revision": LAYA_REV,
               "snapshot": snap}, open(os.path.join(a.out_dir, "classifier_config.json"), "w"), indent=1)


best, step = -1.0, 0
if a.epochs == 0: save(); report["best_epoch"] = 0
for ep in range(1, a.epochs + 1):
    model.train(); t = time.perf_counter(); tot = 0.0
    for k, ids, mask in batches(train, a.batch, True):
        w = torch.tensor([W.get(lab[x].get("confidence"), 0.8) for x in k]); lg = model(ids, mask)
        loss = sum(ce(lg[tk], torch.tensor([TASKS[tk].index(lab[x][tk]) for x in k])) for tk in TASKS)
        loss = (loss * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        tot += float(loss); step += 1
        if step % 50 == 0: print(f"epoch {ep} step {step}/{steps} loss {tot / 50:.3f} {time.perf_counter() - t:.0f}s", flush=True); tot = 0.0
    p = predict(dev + hold); r = {**score(p, dev, lab, "dev"), **score(p, hold, gold, "holdout"), "seconds": round(time.perf_counter() - t)}
    report[f"epoch{ep}"] = r; print(f"epoch {ep}:", r, flush=True)
    dev_acc = sum(r[f"dev_{tk}"] for tk in TASKS if f"dev_{tk}" in r)
    if dev_acc > best: best = dev_acc; report["best_epoch"] = ep; save()
    json.dump(report, open(os.path.join(a.out_dir, "report.json"), "w"), indent=1)
json.dump(report, open(os.path.join(a.out_dir, "report.json"), "w"), indent=1)
print("best epoch", report.get("best_epoch"), report.get(f"epoch{report.get('best_epoch')}"), flush=True)
