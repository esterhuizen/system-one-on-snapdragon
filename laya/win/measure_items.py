"""Measure every question row of Jev-format items with the exact Laya/NPU prompt builder.

python -X utf8 measure_items.py ITEMS.jsonl [--json OUT.json] [--models %USERPROFILE%/local-laya/models/npu]

Per question: untruncated row length L (= what Laya would see without cutting the state), option count,
whether the 192-token question head was cut, and the NPU bucket it routes to. Per item: the tier
  t128 = every row <= 128 tokens            (routes to the 128 NPU buckets)
  t512 = every row in 257..512 tokens       (routes to the 512 NPU bucket, no truncation)
  other = anything else (mixed, 129..256, or > 512 = truncated)
"""
import os
import argparse, json, re, sys
from pathlib import Path
from laya_snapdragon.common import build_prefix, serialize_state
from laya_snapdragon.tokenizer import Tokenizer

ap = argparse.ArgumentParser(); ap.add_argument("items"); ap.add_argument("--json")
ap.add_argument("--models", default=os.path.expandvars(r"%USERPROFILE%\local-laya\models\npu")); a = ap.parse_args()
models = Path(a.models); cfg = json.loads((models / "laya" / "rl_agent_config.json").read_text())
tok = Tokenizer(models / "laya" / "tokenizer"); HEAD = cfg.get("head_max_len", 192)
buckets = sorted({tuple(map(int, re.search(r"_s(\d+)_m(\d+)_", p.name).groups())) for p in (models / "onnx").glob("laya_s*_m*_qnn_ctx.onnx")})

def to_internal(d):   # laya agent.py _to_internal semantics
    t, crit = d["type"], d.get("criteria")
    if t == "choice" and isinstance(crit, list): crit = dict.fromkeys(crit)
    if t == "noul" and isinstance(crit, dict): crit = {str(k).lower(): v for k, v in crit.items()}
    ins = d["instructions"] if isinstance(d["instructions"], str) else json.dumps(d["instructions"], ensure_ascii=False)
    return {"t": t, "ins": ins, "crit": crit}

def head_len_uncut(q):   # prefix length if nothing had been trimmed to fit head_max_len
    ids, _ = build_prefix(tok, q, head_max_len=10_000); return len(ids)

out, summary = [], {}
for line in open(a.items, encoding="utf-8"):
    if not line.strip(): continue
    it = json.loads(line); st_ids = tok(serialize_state(it["state"]).replace(tok.mask_token, " "))["input_ids"]; rows = []
    for qid, d in it["questions"].items():
        q = to_internal(d); ids, markers = build_prefix(tok, q, HEAD)
        L = len(ids) + len(st_ids) + 1; k = len(markers)
        b = next((f"{s}x{m}" for s, m in buckets if L <= s and k <= m), None)
        rows.append({"qid": qid, "type": d["type"], "L": L, "options": k, "head": len(ids),
                     "head_cut": head_len_uncut(q) > len(ids), "npu_bucket": b})
    Ls = [r["L"] for r in rows]
    tier = "t128" if max(Ls) <= 128 else "t512" if min(Ls) > 256 and max(Ls) <= 512 else "other"
    rec = {"id": it["id"], "tier": tier, "state_tokens": len(st_ids), "L_min": min(Ls), "L_max": max(Ls),
           "head_cut": any(r["head_cut"] for r in rows), "no_npu": [r["qid"] for r in rows if not r["npu_bucket"]], "rows": rows}
    out.append(rec); summary[tier] = summary.get(tier, 0) + 1
    print(f"{it['id']:<22} {tier:<6} state={len(st_ids):4d}  L={min(Ls):3d}..{max(Ls):3d}  q={len(rows)}"
          f"{'  HEAD-CUT' if rec['head_cut'] else ''}{'  NO-NPU:' + ','.join(rec['no_npu']) if rec['no_npu'] else ''}")
print("tiers:", summary, "| NPU buckets:", [f"{s}x{m}" for s, m in buckets], file=sys.stderr)
if a.json: Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
