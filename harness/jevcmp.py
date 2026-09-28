"""Send byte-identical state+questions to Jev and local Laya targets; one JSONL line per (repeat, item, target).
python -X utf8 jevcmp.py --items results/items/smoke.jsonl --targets laya-gpu --repeats 3 --out results/runs/smoke/laya-gpu
python -X utf8 jevcmp.py --ping --targets jev,laya-gpu"""
import argparse, hashlib, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
TARGETS = json.loads((Path(__file__).resolve().parent / "targets.json").read_text())
def key(cfg): return os.environ.get(cfg["key_env"], "") if "key_env" in cfg else cfg.get("key", "local")
def qhash(s, q): return hashlib.sha256(json.dumps({"s": s, "q": q}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
def ping(c, name, n=10):
    cfg, ts = TARGETS[name], []
    for _ in range(n):
        t0 = time.perf_counter(); r = c.get(cfg["base"] + cfg["ping"], headers={"Authorization": f"Bearer {key(cfg)}"}, timeout=30)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort(); return {"target": name, "status": r.status_code, "rtt_p50_ms": ts[len(ts) // 2], "rtt_min_ms": ts[0], "body": r.text[:300]}
def call(c, name, state, questions):
    cfg = TARGETS[name]; body = {"state": state, "model": cfg["model"], "questions": questions}
    for attempt in range(6):
        t0 = time.perf_counter()
        try:
            r = c.post(cfg["base"] + "/v1/systemone", json=body, headers={"Authorization": f"Bearer {key(cfg)}"}, timeout=cfg.get("timeout", 120))
        except httpx.HTTPError as e:
            return {"status": -1, "attempt": attempt, "e2e_ms": None, "error": repr(e)[:500]}
        dt = (time.perf_counter() - t0) * 1e3
        if r.status_code in (408, 429, 529) or (r.status_code >= 500 and name.startswith("jev")):
            time.sleep(min(60.0, float(r.headers.get("retry-after", 2 ** attempt)))); continue
        try: j = r.json()
        except ValueError: j = None
        return {"status": r.status_code, "attempt": attempt, "e2e_ms": dt,
                "server_ms": r.headers.get("x-server-time-ms") or r.headers.get("x-inference-time-ms"),
                "backend": r.headers.get("x-laya-backend"), "request_id": r.headers.get("x-typesafe-request-id"),
                "resp": j if isinstance(j, dict) else None, "error": None if r.status_code == 200 else r.text[:1000]}
    return {"status": r.status_code, "attempt": attempt, "e2e_ms": None, "error": "gave up after retries"}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--items"); ap.add_argument("--targets", required=True)
    ap.add_argument("--repeats", type=int, default=1); ap.add_argument("--out"); ap.add_argument("--ping", action="store_true")
    ap.add_argument("--limit", type=int); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--jev-token-budget", type=int, default=2_000_000)
    ap.add_argument("--concurrency", type=int, default=1)
    a = ap.parse_args(); names = a.targets.split(",")
    with httpx.Client() as c:
        if a.ping:
            for n in names: print(json.dumps(ping(c, n)))
            return
        items = [json.loads(l) for l in open(a.items, encoding="utf-8") if l.strip()][: a.limit]
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        for n in names:
            if (out / f"{n}.jsonl").exists(): raise SystemExit(f"{out / (n + '.jsonl')} exists; use a new --out or delete it")
        (out / "meta.json").write_text(json.dumps({"targets": {n: {k: v for k, v in TARGETS[n].items() if k != "key"} for n in names},
            "items": a.items, "repeats": a.repeats, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pings": [ping(c, n, 5) for n in names]}, indent=1))
        files = {n: open(out / f"{n}.jsonl", "w", encoding="utf-8") for n in names}; rng = random.Random(a.seed)
        for n in names:                      # untimed, unrecorded warm-up per local server (plan 9.4)
            if items and not n.startswith("jev"): call(c, n, items[0]["state"], items[0]["questions"])
        lock = threading.Lock(); tok = [0]
        def one(rep, it, n):
            with lock:
                if n.startswith("jev") and tok[0] >= a.jev_token_budget: return
            rec = call(c, n, it["state"], it["questions"]); resp = rec.pop("resp", None) or {}; usage = resp.get("usage") or {}
            line = json.dumps({"rep": rep, "item_id": it["id"], "suite": it.get("suite"), "target": n,
                "qhash": qhash(it["state"], it["questions"]), "model_requested": TARGETS[n]["model"], "n_questions": len(it["questions"]),
                **rec, "resp_model": resp.get("model"), "routing_model": (resp.get("routing") or {}).get("model"),
                "usage": usage, "answers": resp.get("answers")}, ensure_ascii=False) + "\n"
            with lock:
                if n.startswith("jev"): tok[0] += int(usage.get("input_tokens") or 0)
                files[n].write(line); files[n].flush()
        for rep in range(a.repeats):
            if a.concurrency > 1:            # remote targets only: parallel requests (latency then includes queueing effects)
                with ThreadPoolExecutor(a.concurrency) as ex:
                    list(ex.map(lambda args: one(*args), [(rep, it, n) for it in items for n in names]))
                continue
            for it in items:
                order = names[:]; rng.shuffle(order)                  # interleave so drift hits every target equally
                for n in order: one(rep, it, n)
        jev_tok = tok[0]
        print(f"done; jev input tokens {jev_tok} (~${jev_tok * 0.042e-6:.4f})")
if __name__ == "__main__": main()
