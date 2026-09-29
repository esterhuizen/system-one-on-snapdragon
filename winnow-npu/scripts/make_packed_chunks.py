"""Option 2 (both questions in one pass): turn each LPBQ chunk's quantized RoPE tables and attention masks into graph inputs.

cos_/sin_ (swa, glb) and mask_ (swa, glb) are uint16 initializers feeding DequantizeLinear in the calibrated QDQ chunks.
They become uint16 graph inputs of the same shape (the CPU quantizes per-request tables/masks with the chunk's own scale
and zero point), so weights, scales and calibration are untouched. Writes model_packed.onnx next to model.onnx, sharing
its model.data (no copy).   python make_packed_chunks.py [--variant lpbq32]
"""
import argparse, os
import onnx
from onnx import helper as oh

W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="lpbq32"); ap.add_argument("--chain", default="chain"); a = ap.parse_args()
DYN = ("cos_swa", "sin_swa", "cos_glb", "sin_glb", "mask_swa", "mask_glb")
for k in range(12):
    d = os.path.join(W, "onnx", a.chain, f"c{k:02d}_{a.variant}")
    m = onnx.load(os.path.join(d, "model.onnx"), load_external_data=False)
    init = {i.name: i for i in m.graph.initializer}
    made = []
    for base in DYN:
        qn = base + "_quantized"
        if qn not in init:
            continue                                   # a chunk may hold only one layer kind
        t = init[qn]
        m.graph.initializer.remove(t)
        m.graph.input.append(oh.make_tensor_value_info(qn, t.data_type, list(t.dims)))
        made.append(qn)
    onnx.save(m, os.path.join(d, "model_packed.onnx"))   # initializers still reference model.data in this folder
    print(f"chunk {k}: inputs {['x'] + made}", flush=True)
