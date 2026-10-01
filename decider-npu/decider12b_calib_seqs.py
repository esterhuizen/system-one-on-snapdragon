"""Calibration sequences for decider-12b on the NPU, in Decider's own chat-layout prompt format: one sequence per question
(template head + "Context:" + state + question block + template tail + "Answer: ("), built by decider.serve.prepare from
the public synthetic suite. Writes a JSON list of token-id lists (<= 512 tokens).
python decider12b_calib_seqs.py --items suite-v1.jsonl --model-dir DIR --out seqs.json"""
import os, argparse, json
from transformers import AutoTokenizer
from decider import serve as SV
from decider.prompt import chat_template
ap = argparse.ArgumentParser(); ap.add_argument("--items", required=True); ap.add_argument("--model-dir", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
cfg = json.load(open(os.path.join(a.model_dir, "decider_config.json"))); SV.apply_config(cfg)
tok = AutoTokenizer.from_pretrained(a.model_dir); chat = chat_template(tok)
seqs = []
for l in open(a.items, encoding="utf-8"):
    r = json.loads(l)
    rqs, index, items, ctx_len = SV.prepare(tok, r["state"], r["questions"], True, SV.ISOLATED, 32768, chat=chat)
    for it in items:
        assert it["slots"] == [len(it["ids"]) - 1], "answer slot is not the last token"
        if len(it["ids"]) <= 512: seqs.append(it["ids"])
json.dump(seqs, open(a.out, "w")); print(f"{len(seqs)} sequences <= 512 tokens; layout {SV.LAYOUT}; isolated {SV.ISOLATED}; T {SV.TEMP} {SV.TEMP_BY_TYPE}")
