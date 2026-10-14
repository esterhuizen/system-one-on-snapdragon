r"""Plain block-wise int4 layer (DequantizeLinear axis=0 block_size=B -> MatMul, uint16 activations) on the
onnxruntime 1.30 + onnxruntime-qnn 2.6.0 plugin EP, NPU device. Same layer and refs as tools/lpbq_plugin_test.py.
Runs the model twice: enable_block_quant_weight_optimization = 0 (standard BQ path) and 1 (BQ -> LPBQ conversion).
Usage (run env, from winnow-npu):  python tools\w4blk_plugin_test.py [path\to\layer0_s512_w4blk64\model.onnx]
Default model: %WINNOW_HOME%\onnx\layer0_s512_w4blk64\model.onnx (built by tools\phase2_layer.py --layer 0 --mode w4blk).
Follow-up for https://github.com/onnxruntime/onnxruntime-qnn/issues/893 (the maintainer asked for a 2.6.0 re-test)."""
import collections, os, sys, time, numpy as np, onnxruntime as ort, onnxruntime_qnn as oq
W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(W, "onnx", "layer0_s512_w4blk64", "model.onnx")
ort.register_execution_provider_library(oq.EP_NAME, oq.get_library_path())
print("ort", ort.__version__, "qnn plugin", getattr(oq, "__version__", "?"), "libs", oq.get_library_path(), flush=True)
print("model", path, flush=True)
devs = [d for d in ort.get_ep_devices() if d.ep_name == oq.EP_NAME and d.device.type == ort.OrtHardwareDeviceType.NPU]
io = np.load(os.path.join(W, "refs", "layer0_io.npz"))
n = len([k for k in io.files if k.startswith("in")])
samples = [(io[f"in{i}"], io[f"out{i}"]) for i in range(n)]
for bq_opt in ("0", "1"):
    tag = f"[2.6.0 plugin, enable_block_quant_weight_optimization={bq_opt}]"
    so = ort.SessionOptions(); so.log_severity_level = int(os.environ.get("ORT_SEV", "3"))
    so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    so.add_provider_for_devices(devs, {"htp_performance_mode": "burst", "enable_block_quant_weight_optimization": bq_opt})
    t = time.perf_counter()
    try:
        s = ort.InferenceSession(path, so)
    except Exception as e:
        print(tag, "FAIL", str(e)[:600], flush=True); continue
    place = {sg.ep_name: dict(collections.Counter(nd.op_type for nd in sg.get_nodes())) for sg in s.get_provider_graph_assignment_info()}
    print(tag, f"compiled {time.perf_counter() - t:.0f}s placement {place}", flush=True)
    S, H = s.get_inputs()[0].shape[1], s.get_inputs()[0].shape[2]
    rel = []
    for x, y in samples:
        p = np.zeros((1, S, H), np.float32); p[0, :len(x)] = x
        got = s.run(["y"], {"x": p})[0][0, :len(x)]
        rel.append(float(np.linalg.norm(got - y) / np.linalg.norm(y)))
    print(tag, "NPU relative L2 error per sample", [f"{r:.3f}" for r in rel], f"max {max(rel):.3f}", flush=True)
    x0 = samples[0][0]; p = np.zeros((1, S, H), np.float32); p[0, :len(x0)] = x0
    for _ in range(3): s.run(None, {"x": p})
    ts = []
    for _ in range(15):
        t = time.perf_counter(); s.run(None, {"x": p}); ts.append((time.perf_counter() - t) * 1e3)
    print(tag, f"latency {np.median(ts):.1f} ms/layer", flush=True)
