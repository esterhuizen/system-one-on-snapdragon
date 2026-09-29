"""How many w8a16 chunk contexts can be resident and executable at once on the HTP? Load one more, run all loaded."""
import os, time, numpy as np, onnxruntime as ort
W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sess = []
x = np.load(os.path.join(W, "calib", "in_00.npy"), mmap_mode="r")[:1].copy()
for k in range(12):
    so = ort.SessionOptions(); so.log_severity_level = 3
    t = time.perf_counter()
    sess.append(ort.InferenceSession(os.path.join(W, "onnx", "chain", f"c{k:02d}_w8a16", "model_ctx.onnx"), so,
                providers=[("QNNExecutionProvider", {"backend_path": "QnnHtp.dll", "htp_performance_mode": "burst"})]))
    lt = time.perf_counter() - t
    try:
        y = x; t = time.perf_counter()
        for s in sess: y = s.run(["y"], {"x": y})[0]
        print(f"{k + 1} chunks resident: load {lt:.1f}s, run all {(time.perf_counter() - t) * 1e3:.0f} ms OK", flush=True)
    except Exception as e:
        print(f"{k + 1} chunks resident: FAILED {str(e)[-80:]}", flush=True); break
