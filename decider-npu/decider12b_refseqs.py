"""Six public JevBench requests -> decider-12b rows (Decider's chat-layout prompt via decider.serve.prepare): token ids,
option count and answer type per row, for the PyTorch reference and the NPU check."""
import os, json
from transformers import AutoTokenizer
from decider import serve as SV
from decider.prompt import chat_template, letter_ids
MD = os.path.join(os.environ.get("DECIDER_MODELS") or "models", "decider-12b")
SV.apply_config(json.load(open(MD + r"\decider_config.json"))); tok = AutoTokenizer.from_pretrained(MD); chat = chat_template(tok)
J = os.environ.get("JEVBENCH_PUBLIC", os.path.join("jevbench", "datasets", "public"))
pick = [json.loads(l) for l in open(os.path.join(J, "easy.jsonl"), encoding="utf-8")][:2] + [json.loads(l) for l in open(os.path.join(J, "original.jsonl"), encoding="utf-8")][:3]
hard = [json.loads(l) for l in open(os.path.join(J, "hard.jsonl"), encoding="utf-8")]; pick.append(next(h for h in hard if len(json.dumps(h["state"])) < 1200))
out = []
for it in pick:
    q = {"type": it["question"]["type"], "instructions": it["question"]["instructions"]}
    if it["question"].get("criteria") is not None: q["criteria"] = it["question"]["criteria"]
    body = {"state": it["state"], "questions": {"decision": q}}
    rqs, index, items, ctx_len = SV.prepare(tok, body["state"], body["questions"], True, SV.ISOLATED, 32768, chat=chat)
    for r in items:
        out.append({"id": it["id"], "body": body, "ids": r["ids"], "n": r["nopts"][0], "type": r["types"][0], "expected": it["expected"]})
json.dump({"letters": letter_ids(tok)[:64], "rows": out}, open(os.path.join(os.environ.get("DECIDER_MODELS") or "models", "decider12b_refrows.json"), "w"))
print(len(out), "rows, lengths", [len(r["ids"]) for r in out])
