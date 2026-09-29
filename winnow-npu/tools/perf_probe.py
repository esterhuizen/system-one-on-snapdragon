"""Sustained NPU chain latency: default run options vs keeping the HTP in burst between runs (qnn.htp_perf_mode*)."""
import os, time, numpy as np, onnxruntime as ort
W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
x0 = np.load(os.path.join(W, "calib", "in_00.npy"), mmap_mode="r")[:1].copy()
sess = []
import sys
NCH = int(sys.argv[1]) if len(sys.argv) > 1 else 12
OFF = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for k in range(OFF, OFF + NCH):
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess.append(ort.InferenceSession(os.path.join(W, "onnx", "chain", f"c{k:02d}_w8a16", "model_ctx.onnx"), so,
                providers=[("QNNExecutionProvider", {"backend_path": "QnnHtp.dll", "htp_performance_mode": "burst"})]))
def passes(ro, n=int(os.environ.get("PASSES", "8"))):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); y = x0
        for s in sess: y = s.run(["y"], {"x": y}, ro)[0]
        ts.append((time.perf_counter() - t) * 1e3)
    return f"median {np.median(ts):.0f} ms  all {[round(v) for v in ts]}"
print(f"{NCH} chunks from {OFF}, default     :", passes(None), flush=True)
ro = ort.RunOptions(); ro.add_run_config_entry("qnn.htp_perf_mode", "burst"); ro.add_run_config_entry("qnn.htp_perf_mode_post_run", "burst")
print(f"{NCH} chunks, burst pre+post:", passes(ro), flush=True)
