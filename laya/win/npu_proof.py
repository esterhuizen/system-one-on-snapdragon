"""Prove a compiled Laya NPU bucket executes on the Hexagon HTP; parity + latency vs ORT CPU on the same static graph.
python -X utf8 npu_proof.py --models %USERPROFILE%/local-laya/models/npu --seq 128 [--profile] [--adsp]"""
import argparse, collections, importlib.util, os, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--models", default=os.path.expandvars(r"%USERPROFILE%\local-laya\models\npu")); ap.add_argument("--seq", type=int, default=128)
ap.add_argument("--markers", type=int, default=8); ap.add_argument("--profile", action="store_true"); ap.add_argument("--adsp", action="store_true")
ap.add_argument("-n", type=int, default=200); a = ap.parse_args()
spec = importlib.util.find_spec("onnxruntime_qnn")
qdir = Path(spec.origin).parent if spec else Path(importlib.util.find_spec("onnxruntime").origin).parent / "capi"
if a.adsp:
    os.environ["ADSP_LIBRARY_PATH"] = str(qdir) + (";" + os.environ["ADSP_LIBRARY_PATH"] if os.environ.get("ADSP_LIBRARY_PATH") else "")
import numpy as np, onnx, onnxruntime as ort
d = Path(a.models) / "onnx"
ctx, src = d / f"laya_s{a.seq}_m{a.markers}_qnn_ctx.onnx", d / f"laya_s{a.seq}_m{a.markers}.onnx"
builtin = "QNNExecutionProvider" in ort.get_available_providers()
print("onnxruntime", ort.__version__, "| QNN", "built-in (1.x)" if builtin else "plugin (2.x)", "| libs", qdir, flush=True)
opts = {"htp_performance_mode": "burst", "enable_htp_fp16_precision": "1"}
if a.profile:
    opts.update(profiling_level="basic", profiling_file_path=str(d / f"qnn_profile_s{a.seq}.csv"))

def qnn_session(path):
    so = ort.SessionOptions(); so.log_severity_level = 1   # INFO: backend/device, 'Switching to user driver path' if HNRD
    so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    if builtin:
        return ort.InferenceSession(str(path), so, providers=[("QNNExecutionProvider", {"backend_path": "QnnHtp.dll", **opts}), "CPUExecutionProvider"])
    import onnxruntime_qnn as oq
    ort.register_execution_provider_library(oq.EP_NAME, oq.get_library_path())
    devs = [x for x in ort.get_ep_devices() if x.ep_name == oq.EP_NAME and x.device.type == ort.OrtHardwareDeviceType.NPU]
    assert devs, "QNN EP exposes no NPU device"
    so.add_provider_for_devices(devs, opts); return ort.InferenceSession(str(path), so)

print("[ctx graph]", dict(collections.Counter(n.op_type for n in onnx.load(str(ctx), load_external_data=False).graph.node)),
      "(EPContext nodes can only be executed by the QNN EP)", flush=True)
t0 = time.perf_counter(); npu = qnn_session(ctx); print(f"[load] {time.perf_counter()-t0:.1f}s providers={npu.get_providers()}", flush=True)
try:
    for sg in npu.get_provider_graph_assignment_info():
        print(f"[placement] {sg.ep_name}: {dict(collections.Counter(n.op_type for n in sg.get_nodes()))}", flush=True)
except AttributeError:
    print("[placement] API unavailable in this build; rely on [ctx graph] + latency", flush=True)
cpu = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0); L = a.seq - 16
ids = np.full((1, a.seq), 50283, np.int64); ids[0, :L] = rng.integers(5, 30000, L); ids[0, 0] = 50281
att = np.zeros((1, a.seq), np.int64); att[0, :L] = 1
feed = {"input_ids": ids, "attention_mask": att, "marker_pos": (np.arange(a.markers, dtype=np.int64) * 3 + 10)[None],
        "marker_mask": np.array([[True] * 4 + [False] * (a.markers - 4)]), "qtype": np.array([0], np.int64)}
ref, got = cpu.run(["logits"], feed)[0][0, :4], npu.run(["logits"], feed)[0][0, :4]
print(f"[parity] max|dlogit| NPU vs CPU {np.abs(ref - got).max():.4f} argmax equal {ref.argmax() == got.argmax()}", flush=True)

def bench(s, n):
    for _ in range(10): s.run(None, feed)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); s.run(None, feed); ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts); return f"p50 {np.percentile(ts,50):.1f} ms p95 {np.percentile(ts,95):.1f} ms min {ts.min():.1f} ms n={n}"
print("[latency] NPU", bench(npu, a.n), flush=True); print("[latency] CPU", bench(cpu, 30), flush=True)
if a.profile: print("[profile]", opts["profiling_file_path"])
