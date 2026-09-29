"""Compile the 12 packed LPBQ chunks for the Hexagon NPU once and cache the QNN contexts (model_packed<S>_ctx.onnx + .bin).

Run in the RUN environment (onnxruntime 1.30 + onnxruntime-qnn 2.6.0 plugin; the older 1.24.x EP rejects LPBQ weights).
Compile in its own process and start the server afterwards: running straight after compiling in the same process failed
with QNN error 1003 here.        python compile_chain.py [--chain chain_pub] [--seq 576]
"""
import argparse, os, time
import onnxruntime as ort
import onnxruntime_qnn as oq

W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser(); ap.add_argument("--chain", default="chain_pub"); ap.add_argument("--seq", type=int, default=576)
ap.add_argument("--variant", default="lpbq32"); a = ap.parse_args()
mp = "model_packed" if a.seq == 512 else f"model_packed{a.seq}"
ort.register_execution_provider_library(oq.EP_NAME, oq.get_library_path())
npu = [x for x in ort.get_ep_devices() if x.ep_name == oq.EP_NAME and x.device.type == ort.OrtHardwareDeviceType.NPU]
assert npu, "no QNN NPU device"
for k in range(12):
    d = os.path.join(W, "onnx", a.chain, f"c{k:02d}_{a.variant}"); src = os.path.join(d, mp + ".onnx"); ctx = os.path.join(d, mp + "_ctx.onnx")
    if os.path.exists(ctx):
        print(f"chunk {k}: cached"); continue
    so = ort.SessionOptions(); so.log_severity_level = 3
    so.add_session_config_entry("ep.context_enable", "1"); so.add_session_config_entry("ep.context_file_path", ctx)
    so.add_session_config_entry("ep.context_embed_mode", "0")
    so.add_provider_for_devices(npu, {"htp_performance_mode": "burst", "htp_graph_finalization_optimization_mode": "3"})
    t = time.perf_counter(); s = ort.InferenceSession(src, so); del s
    print(f"chunk {k}: compiled in {time.perf_counter() - t:.0f}s -> {ctx}", flush=True)
