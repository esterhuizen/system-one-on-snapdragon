"""Jev-compatible HTTP server for Winnow-12B on the Hexagon NPU (npu_winnow.WinnowNPU). 127.0.0.1 only, no auth.

    POST /v1/systemone {"state": ..., "questions": {...}}   GET /health   GET /v1/models
python npu_serve.py [--port 8014] [--chain chain_pub] [--seq 576]
"""
import argparse, json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8014)
ap.add_argument("--chain", default="chain_pub"); ap.add_argument("--seq", type=int, default=576)
a = ap.parse_args()
try:
    import nothrottle  # noqa: F401  (opt out of Windows EcoQoS for a hidden background server, if the helper is present)
except ImportError:
    pass
from npu_winnow import WinnowNPU  # noqa: E402
t = time.perf_counter(); RT = WinnowNPU(chain=a.chain, seq=a.seq)
print(f"Winnow NPU runtime ready in {time.perf_counter() - t:.0f}s (chain {a.chain}, S={a.seq})", flush=True)


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj, extra=None):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health": return self._send(200, {"ok": True, "model": "Winnow-12B-npu-lpbq", "seq": a.seq, "chain": a.chain})
        if self.path == "/v1/models": return self._send(200, {"models": [{"name": "Winnow-12B-npu-lpbq"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/systemone": return self._send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            out = RT.decide(body)
            self._send(200, out, {"x-server-time-ms": f"{out['npu']['ms']:.1f}"})
        except (ValueError, KeyError, TypeError) as e:
            self._send(422, {"error": f"{type(e).__name__}: {e}"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *_):
        pass


print(f"serving on 127.0.0.1:{a.port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
