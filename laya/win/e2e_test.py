"""End-to-end: stock laya (torch CPU) vs laya with encoder on ORT GPU EP; same questions; compare answers + latency."""
import os, sys, time, copy, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laya
from laya_ort_encoder import to_gpu

model_dir, backend = sys.argv[1], sys.argv[2]
TAG = sys.argv[3] if len(sys.argv) > 3 else "fp32g"
state = {"from": "user@acme.com", "subject": "Duplicate charge on invoice #4411",
         "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."}
questions = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                                "sales": "pricing, new contracts", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    "refund_requested": {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
}
long_state = {"body": " ".join(["The deployment pipeline failed again after the database migration and customers cannot log in."] * 20)}

cpu = laya.load("convaiinnovations/laya", device="cpu")
gpu = to_gpu(laya.load("convaiinnovations/laya", device="cpu"), model_dir, backend=backend, tag=TAG)

def summarize(r):
    out = {}
    for k, a in r["answers"].items():
        v = a.get("choice", a.get("score", a.get("noul")))
        out[k] = (v, {kk: round(vv, 3) for kk, vv in a.get("probabilities", {}).items()} if "probabilities" in a else None)
    return out

for name, st in [("readme_ticket", state), ("long_state", long_state)]:
    for label, ag in [("torch-cpu", cpu), (backend, gpu)]:
        ag.predict(st, questions)  # warm
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); r = ag.predict(st, questions); ts.append(time.perf_counter() - t0)
        ts.sort()
        print(f"{name:14s} {label:10s} tokens={r['usage']['input_tokens']:5d} median={ts[2]*1e3:7.1f} ms  {json.dumps(summarize(r))}", flush=True)
