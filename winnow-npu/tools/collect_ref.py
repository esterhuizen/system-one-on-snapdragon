"""Collect reference outputs from the CPU Winnow server (Q8_0): exact token ids (/v1/winnow/inspect) and raw answer
logits (/v1/systemone with winnow.diagnostics) for a few requests. These are the parity targets for the PyTorch rebuild
and for the NPU export.   python collect_ref.py --out refs\\ref.json [--base http://127.0.0.1:8013]
"""
import argparse, json, os
import httpx

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--base", default="http://127.0.0.1:8013")
a = ap.parse_args()
J = os.environ.get("JEVBENCH_PUBLIC", os.path.join("jevbench", "datasets", "public"))   # clone of fstandhartinger/jevbench


def jb(file, n):
    rows = [json.loads(l) for l in open(os.path.join(J, file), encoding="utf-8")]
    out = []
    for t in rows:
        if len(out) == n: break
        q = {"type": t["question"]["type"], "instructions": t["question"]["instructions"]}
        if t["question"].get("criteria") is not None: q["criteria"] = t["question"]["criteria"]
        if len(json.dumps(t["state"])) < 1500:
            out.append((t["id"], {"state": t["state"], "questions": {"decision": q}}))
    return out


reqs = jb("easy.jsonl", 1) + jb("original.jsonl", 2)
hard = [json.loads(l) for l in open(os.path.join(J, "hard.jsonl"), encoding="utf-8")]
noul = next(t for t in hard if t["question"]["type"] == "noul" and len(json.dumps(t["state"])) < 2000)
reqs.append((noul["id"], {"state": noul["state"], "questions": {"decision": {k: v for k, v in noul["question"].items() if k in ("type", "instructions", "criteria") and v is not None}}}))
reqs.append(("synthetic-ticket", {"state": {"summary": "Outlook keeps asking for my password after the update", "request_type": "Incident"},
                                    "questions": {"queue": {"type": "choice", "instructions": "Which team should handle this ticket?",
                                                            "criteria": {"accounts": "sign-in, passwords, MFA", "email": "mailboxes, Outlook", "devices": "laptops, printers", "network": "Wi-Fi, VPN"}},
                                                  "urgent": {"type": "noul", "instructions": "Does this need attention today?"}}}))
c = httpx.Client(timeout=600)
out = []
for rid, body in reqs:
    body = dict(body, model="Winnow-12B")
    ins = c.post(a.base + "/v1/winnow/inspect", json=dict(body, winnow={"include_token_ids": True})).json()
    ans = c.post(a.base + "/v1/systemone", json=dict(body, winnow={"diagnostics": True})).json()
    out.append({"id": rid, "request": body, "inspect": ins, "answers": ans.get("answers"), "error": ans.get("error")})
    print(rid, "prefix", len(ins.get("prefix_token_ids", [])), "suffixes", [len(s) for s in ins.get("suffix_token_ids", [])],
          {k: (v.get("winnow", {}).get("logits"), v.get("choice", v.get("noul"))) for k, v in (ans.get("answers") or {}).items()}, flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
