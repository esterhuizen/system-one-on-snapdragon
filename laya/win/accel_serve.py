"""Jev-compatible POST /v1/systemone serving Laya *english* on one backend (laya 0.3.20 Router + create_app).
 npu: laya_snapdragon Agent(device='npu') strict (no silent CPU; non-fitting question -> 422) | npu-auto | ortcpu (ORT CPU fp32)
 qnngpu / dml: stock laya torch Agent with the encoder swapped to static ONNX buckets on the Adreno.
python -X utf8 accel_serve.py --backend qnngpu --port 8001 [--adsp]"""
import argparse, importlib.util, os, sys, time
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nothrottle  # noqa: F401  (opt out of Windows EcoQoS before any heavy work)
ROOT = os.path.expandvars(r"%USERPROFILE%\local-laya")
ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True, choices=["npu", "npu-auto", "ortcpu", "qnngpu", "dml"])
ap.add_argument("--port", type=int, required=True); ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--npu-models", default=ROOT + r"\models\npu"); ap.add_argument("--enc-dir", default=ROOT + r"\models\gpu-enc")
ap.add_argument("--tag", default="fp32g"); ap.add_argument("--adsp", action="store_true")
a = ap.parse_args()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if a.adsp:  # prepend the dir that ships libQnnHtpV73Skel.so to the fastRPC search path (onnxruntime-qnn PR #593 workaround)
    s = importlib.util.find_spec("onnxruntime_qnn")
    d = Path(s.origin).parent if s else Path(importlib.util.find_spec("onnxruntime").origin).parent / "capi"
    os.environ["ADSP_LIBRARY_PATH"] = str(d) + (";" + os.environ["ADSP_LIBRARY_PATH"] if os.environ.get("ADSP_LIBRARY_PATH") else "")
import uvicorn
from laya.router import Router, normalise_name
from laya.serve import create_app

class EnglishOnlyRouter(Router):
    """Never auto-route to / load a torch multilingual or typed-decisions checkpoint behind an accelerator."""
    def load(self, name):
        if normalise_name(name) != "english":
            raise ValueError(f"this server only serves the 'english' checkpoint on {a.backend}; send \"model\": \"english\"")
        return super().load(name)

router = EnglishOnlyRouter(lang_guess=lambda state: "en")
if a.backend in ("npu", "npu-auto", "ortcpu"):
    from laya_snapdragon import Agent as SnapAgent
    snap = SnapAgent(a.npu_models, device={"npu": "npu", "npu-auto": "auto", "ortcpu": "cpu"}[a.backend],
                     cpu_threads=int(os.environ.get("LAYA_THREADS", "12")))   # match torch-cpu reference (cpu_serve --threads 12)
    # Reproduce laya 0.3.20 behaviours laya_snapdragon lacks, so npu/ortcpu differ from torch-cpu only by hardware.
    import json, laya_snapdragon as _ls
    _clamp = lambda t: min(5.0, max(0.5, float(t)))                          # laya common.clamp_temperature
    snap.temperature = [_clamp(t) for t in snap.temperature]
    snap.temperature_by_options = {k: _clamp(v) for k, v in snap.temperature_by_options.items()}
    _bs = _ls.build_sequence                                                  # laya agent.py:568 left-truncates list states
    _ls.build_sequence = lambda tok, st, q, *p, **k: _bs(tok, st, q, *p, **{**k, "truncate_left": isinstance(st, list)})
    class SnapAdapter:
        def system_one(self, state, questions, **_ignored):   # lang/max_len overrides unsupported
            fixed = {}
            for qid, d in questions.items():
                if not isinstance(d, dict) or d.get("type") not in ("choice", "score", "noul") or "instructions" not in d:
                    raise ValueError(f"question {qid!r}: invalid definition")
                if "labels" in d:
                    raise ValueError(f"question {qid!r}: noul 'labels' unsupported on {a.backend}")
                if not isinstance(d["instructions"], str):                 # laya agent.py _to_internal: ensure_ascii=False
                    d = {**d, "instructions": json.dumps(d["instructions"], ensure_ascii=False)}
                if d["type"] == "noul" and isinstance(d.get("criteria"), dict):   # laya lower-cases noul criteria keys
                    d = {**d, "criteria": {str(k).lower(): v for k, v in d["criteria"].items()}}
                fixed[qid] = d
            r = snap.predict(state, fixed)
            for ans in r["answers"].values():                 # match laya 0.3.20 torch output
                p = [ans["noul"], 1 - ans["noul"]] if ans["type"] == "noul" else list(ans["probabilities"].values())
                ans["answer_confidence"] = round(max(p), 4)
            return r
        predict = system_one
    router.attach("english", SnapAdapter())
    print(f"[accel_serve] {a.backend}: NPU buckets {[(s_, m_) for s_, m_, _ in snap.buckets]} loaded in {snap.load_s:.1f}s", flush=True)
else:
    import laya
    from laya_ort_encoder import to_gpu
    agent = to_gpu(laya.load("convaiinnovations/laya", device="cpu"), a.enc_dir, backend=a.backend, tag=a.tag)
    router.attach("english", agent)
    print(f"[accel_serve] {a.backend}: encoder buckets {agent.model.encoder.buckets}", flush=True)
app = create_app(router)

@app.middleware("http")
async def _server_time(request, call_next):
    t0 = time.perf_counter(); resp = await call_next(request)
    resp.headers["X-Server-Time-Ms"] = f"{(time.perf_counter() - t0) * 1e3:.2f}"; resp.headers["X-Laya-Backend"] = a.backend
    return resp
uvicorn.run(app, host=a.host, port=a.port, log_level="info")
