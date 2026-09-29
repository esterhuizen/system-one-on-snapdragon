"""LPBQ int4 on the Phase-2 single layer (layer 0, S=512): CPU-simulated error, NPU placement/error/latency per variant."""
import collections, os, sys, time
import numpy as np
import onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpbq import convert  # noqa: E402

W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); L = int(sys.argv[1]) if len(sys.argv) > 1 else 0
src = os.path.join(W, "onnx", f"layer{L}_s512_w8a16", "model.onnx")
io = np.load(os.path.join(W, "refs", f"layer{L}_io.npz"))
xs, ys, lens = [], [], []
for i in range(len([k for k in io.files if k.startswith("in")])):
    x = io[f"in{i}"]; p = np.zeros((1, 512, 3840), np.float32); p[0, :len(x)] = x; xs.append(p); ys.append(io[f"out{i}"]); lens.append(len(x))
err = lambda s: max(np.linalg.norm(s.run(["y"], {"x": x})[0][0, :m] - y) / np.linalg.norm(y) for x, y, m in zip(xs, ys, lens))
variants = [(64, True), (64, False), (32, True)]
for B, mse in variants:
    tag = f"lpbq{B}{'_mse' if mse else ''}"; dst = os.path.join(W, "onnx", f"layer{L}_s512_{tag}", "model.onnx")
    if not os.path.exists(dst):
        convert(src, dst, B, mse)
    cpu = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]); e_cpu = err(cpu); del cpu
    so = ort.SessionOptions(); so.log_severity_level = 3; so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    t = time.perf_counter()
    npu = ort.InferenceSession(dst, so, providers=[("QNNExecutionProvider", {"backend_path": "QnnHtp.dll", "htp_performance_mode": "burst"}), "CPUExecutionProvider"])
    ct = time.perf_counter() - t
    place = {sg.ep_name: sum(1 for _ in sg.get_nodes()) for sg in npu.get_provider_graph_assignment_info()}
    cpu_ops = collections.Counter(n.op_type for sg in npu.get_provider_graph_assignment_info() if sg.ep_name == "CPUExecutionProvider" for n in sg.get_nodes())
    e_npu = err(npu)
    for _ in range(3): npu.run(None, {"x": xs[0]})
    ts = []
    for _ in range(15):
        t = time.perf_counter(); npu.run(None, {"x": xs[0]}); ts.append((time.perf_counter() - t) * 1e3)
    print(f"[{tag}] CPU-sim rel err {e_cpu:.3f} | NPU rel err {e_npu:.3f} | {np.median(ts):.1f} ms/layer (x48 ~{np.median(ts) * 48 / 1e3:.2f} s) | "
          f"compile {ct:.0f}s | placement {place} cpu ops {dict(cpu_ops)} | size {os.path.getsize(dst.replace('model.onnx', 'model.data')) / 2**20:.0f} MB", flush=True)
    del npu
