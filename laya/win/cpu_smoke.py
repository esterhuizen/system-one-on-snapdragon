"""CPU reference smoke test: stock laya 0.3.20 on PyTorch CPU (upstream-exact)."""
import os
import json, time
import laya
t0 = time.perf_counter(); a = laya.load("convaiinnovations/laya", device="cpu"); print(f"load {time.perf_counter()-t0:.1f}s", flush=True)
items = [json.loads(l) for l in open(os.path.expandvars(r"%USERPROFILE%\local-laya\results\items\smoke.jsonl"), encoding="utf-8")]
for it in items[:2]:
    a.predict(it["state"], it["questions"])            # warm-up
    t0 = time.perf_counter(); r = a.predict(it["state"], it["questions"]); dt = (time.perf_counter() - t0) * 1e3
    print(f"[{it['id']}] {dt:.0f} ms  gold={it['gold']}")
    print(json.dumps(r, indent=1, ensure_ascii=False)[:1500])
