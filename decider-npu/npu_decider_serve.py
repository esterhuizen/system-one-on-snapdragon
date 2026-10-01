"""Jev-compatible HTTP server for decider-12b on the Hexagon NPU (npu_decider.DeciderNPU). 127.0.0.1 only, no auth.

    POST /v1/systemone {"state": ..., "questions": {...}}   GET /health   GET /v1/models
python npu_serve.py [--port 8014] [--chain chain_pub] [--seq 576]
"""
import argparse, json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8016)
ap.add_argument("--chain", default="chain_decider12b"); ap.add_argument("--seq", type=int, default=576)
ap.add_argument("--perf", default="burst", help="QNN htp_performance_mode (burst | sustained_high_performance | high_performance | ...)")
a = ap.parse_args()
try:
    import nothrottle  # noqa: F401  (opt out of Windows EcoQoS for a hidden background server, if the helper is present)
except ImportError:
    pass
from npu_decider import DeciderNPU  # noqa: E402
t = time.perf_counter(); RT = DeciderNPU(chain=a.chain, seq=a.seq, perf=a.perf)
print(f"decider-12b NPU runtime ready in {time.perf_counter() - t:.0f}s (chain {a.chain}, S={a.seq}, perf {a.perf})", flush=True)

# Start-up self-test. A process that loads its contexts while a previous NPU process is still releasing its memory can come up
# "healthy" and return garbage: a whole benchmark run came back at chance level with no error. Refuse to serve unless
# two obvious decisions come out right and confident.
CANARY = [({"state": "Please close my account, I no longer want to bank with you.",
            "questions": {"q": {"type": "choice", "instructions": "What does the customer want?",
                                "criteria": {"open_account": None, "close_account": None, "order_card": None, "report_fraud": None}}}}, "close_account"),
          ({"state": "My new card arrived today and works fine, thanks!",
            "questions": {"q": {"type": "choice", "instructions": "Is this a complaint?", "criteria": {"yes": None, "no": None}}}}, "no")]
def self_test(rt):
    for body, want in CANARY:
        ans = rt.decide(body)["answers"]["q"]
        if ans["choice"] != want or ans["probabilities"][want] < 0.8:
            print(f"SELF-TEST FAILED: expected {want!r}, got {ans['choice']!r} p={ans['probabilities'].get(want, 0):.2f}", flush=True)
            return False
    return True


if not self_test(RT):
    raise SystemExit(3)
print("self-test passed", flush=True)


def recover():
    """The NPU can crash mid-run (QNN reports 'SSR detected', e.g. across sleep/resume). Reload every session, self-test,
    and only then carry on; exit rather than serve answers from a runtime that failed its self-test."""
    global RT
    print("NPU error: reloading sessions", flush=True)
    RT = None
    import gc; gc.collect(); time.sleep(5)
    RT = DeciderNPU(chain=a.chain, seq=a.seq, perf=a.perf)
    if not self_test(RT):
        print("recovery failed the self-test; exiting", flush=True); os._exit(4)
    print("recovered", flush=True)


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj, extra=None):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health": return self._send(200, {"ok": True, "model": "decider-12b-npu-lpbq", "seq": a.seq, "chain": a.chain})
        if self.path == "/v1/models": return self._send(200, {"models": [{"name": "decider-12b-npu-lpbq"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/systemone": return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                out = RT.decide(body)
            except (ValueError, KeyError, TypeError):
                raise
            except Exception as e:  # noqa: BLE001  - NPU/QNN failure: recover once and retry this request
                print(f"decide failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
                recover(); out = RT.decide(body)
            self._send(200, out, {"x-server-time-ms": f"{out['npu']['ms']:.1f}"})
        except (ValueError, KeyError, TypeError) as e:
            self._send(422, {"error": f"{type(e).__name__}: {e}"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *_):
        pass


# Single-threaded on purpose: every NPU call must come from the thread that created the QNN sessions. With a threading server
# (one new thread per request) the NPU silently returned wrong answers after ~60-120 requests with no error; the same work
# on one thread ran clean. Requests are served one at a time (the NPU runs one pass at a time anyway).
print(f"serving on 127.0.0.1:{a.port}", flush=True)
HTTPServer(("127.0.0.1", a.port), H).serve_forever()
