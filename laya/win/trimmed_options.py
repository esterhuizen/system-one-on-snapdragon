"""Write exactly what Laya's prompt builder keeps of each option of a choice question (192-token head budget) to JSON."""
import os
import json, sys
from laya_snapdragon.common import build_prefix
from laya_snapdragon.tokenizer import Tokenizer
tok = Tokenizer(os.path.expandvars(r"%USERPROFILE%\local-laya\models\npu\laya\tokenizer"))
tax = json.load(open(sys.argv[1], encoding="utf-8")); out = {}
for qkey, crit_key in (("bucket", "buckets"), ("work_type", "work_types")):
    crit = tax[crit_key]; q = {"t": "choice", "ins": tax["instructions"][qkey], "crit": crit}
    ids, markers = build_prefix(tok, q, 192)
    dec = lambda xs: tok.backend.decode(xs, skip_special_tokens=False)
    seen = {}
    for i, (k, m) in enumerate(zip(crit, markers)):
        end = markers[i + 1] if i + 1 < len(markers) else len(ids) - 1
        seen[k] = dec(ids[m + 1:end]).strip()
    out[qkey] = seen
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(out["bucket"], indent=1, ensure_ascii=False))
