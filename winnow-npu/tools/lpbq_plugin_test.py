"""Same LPBQ layer, but with onnxruntime 1.30 + onnxruntime-qnn 2.6.0 plugin EP (newer QNN SDK), on the NPU device."""
import collections, os, sys, time, numpy as np, onnxruntime as ort, onnxruntime_qnn as oq
W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); path = sys.argv[1]
ort.register_execution_provider_library(oq.EP_NAME, oq.get_library_path())
print("ort", ort.__version__, "qnn plugin", getattr(oq, "__version__", "?"), "libs", oq.get_library_path(), flush=True)
devs = [d for d in ort.get_ep_devices() if d.ep_name == oq.EP_NAME and d.device.type == ort.OrtHardwareDeviceType.NPU]
so = ort.SessionOptions(); so.log_severity_level = int(os.environ.get("ORT_SEV", "3")); so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
so.add_provider_for_devices(devs, {"htp_performance_mode": "burst"})
io = np.load(os.path.join(W, "refs", "layer0_io.npz")); x = io["in5"]; y = io["out5"]; p = np.zeros((1, 512, 3840), np.float32); p[0, :len(x)] = x
t = time.perf_counter()
try:
    s = ort.InferenceSession(path, so)
except Exception as e:
    print("FAIL", str(e)[:400]); raise SystemExit(1)
print(f"compiled {time.perf_counter() - t:.0f}s placement", {sg.ep_name: dict(collections.Counter(n.op_type for n in sg.get_nodes())) for sg in s.get_provider_graph_assignment_info()}, flush=True)
got = s.run(["y"], {"x": p})[0][0, :len(x)]; print("NPU rel err", np.linalg.norm(got - y) / np.linalg.norm(y), flush=True)
ts = []
for _ in range(15):
    t = time.perf_counter(); s.run(None, {"x": p}); ts.append((time.perf_counter() - t) * 1e3)
print(f"latency {np.median(ts):.1f} ms/layer", flush=True)
