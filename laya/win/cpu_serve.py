"""Stock laya 0.3.20 PyTorch-CPU behind laya-serve's own app, loopback only, inter-op threads pinned to 1.
python -X utf8 cpu_serve.py --port 8000 --threads 12 [--preload english,typed-decisions]"""
import argparse, os, time
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nothrottle  # noqa: F401  (opt out of Windows EcoQoS before any heavy work)
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000); ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--preload", default="english"); a = ap.parse_args()
os.environ.update(LAYA_DEVICE="cpu", LAYA_MODELS=a.preload, LAYA_PRELOAD="1", LAYA_THREADS=str(a.threads), HF_HUB_DISABLE_SYMLINKS_WARNING="1")
import torch
torch.set_num_interop_threads(1)      # laya-serve pins intra-op only (BENCHMARKS.md: 9.4 s -> 0.78 s needs both)
import uvicorn
from laya.serve import create_app
app = create_app()                    # Router(device='cpu'), preloads LAYA_MODELS, applies LAYA_THREADS

@app.middleware("http")
async def _server_time(request, call_next):
    t0 = time.perf_counter(); resp = await call_next(request)
    resp.headers["X-Server-Time-Ms"] = f"{(time.perf_counter() - t0) * 1e3:.2f}"; resp.headers["X-Laya-Backend"] = "torch-cpu"
    return resp
uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="info")
