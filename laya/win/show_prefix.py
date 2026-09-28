"""Decode exactly what Laya's prompt builder keeps of a question + options (192-token head budget)."""
import os
import json, sys
from laya_snapdragon.common import build_prefix
from laya_snapdragon.tokenizer import Tokenizer
tok = Tokenizer(os.path.expandvars(r"%USERPROFILE%\local-laya\models\npu\laya\tokenizer"))
crit = json.loads(sys.argv[1]); q = {"t": "choice", "ins": sys.argv[2], "crit": crit}
ids, markers = build_prefix(tok, q, 192)
full, _ = build_prefix(tok, q, 10_000)
dec = lambda xs: tok.backend.decode(xs, skip_special_tokens=False)
print(f"head tokens kept {len(ids)} of {len(full)} needed")
print("QUESTION AS SEEN:", dec(ids[1:markers[0]]).replace("[SEP]", "").strip())
for i, m in enumerate(markers):
    end = markers[i + 1] if i + 1 < len(markers) else len(ids) - 1
    print("  OPTION AS SEEN:", dec(ids[m + 1:end]).strip())
