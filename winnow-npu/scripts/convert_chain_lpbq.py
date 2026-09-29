"""Convert the 12 w8a16 chain chunks to LPBQ int4 (block 32, MSE clip): onnx\\chain\\cXX_w8a16 -> onnx\\chain\\cXX_lpbq32."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lpbq import convert  # noqa: E402
W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CH = sys.argv[1] if len(sys.argv) > 1 else "chain"
for k in range(12):
    src = os.path.join(W, "onnx", CH, f"c{k:02d}_w8a16", "model.onnx"); dst = os.path.join(W, "onnx", CH, f"c{k:02d}_lpbq32", "model.onnx")
    if os.path.exists(dst): print(f"chunk {k} exists"); continue
    t = time.perf_counter(); n = convert(src, dst, 32, True)
    print(f"chunk {k}: {n} weights -> LPBQ32 in {time.perf_counter() - t:.0f}s, {os.path.getsize(dst.replace('model.onnx', 'model.data')) / 2**30:.2f} GB", flush=True)
