"""Where does the w8a16 layer lose accuracy? Run fp32 and QDQ models on ORT CPU, compare every activation (SQNR, dB)."""
import os, sys, numpy as np, onnxruntime as ort
from onnxruntime.quantization.qdq_loss_debug import collect_activations, compute_activation_error, create_activation_matching, modify_model_output_intermediate_tensors
from onnxruntime.quantization import CalibrationDataReader
W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); L = int(sys.argv[1]); tag = sys.argv[2]
fp = os.path.join(W, "onnx", f"layer{L}_s512_fp32", "model.onnx"); q = os.path.join(W, "onnx", f"layer{L}_s512_{tag}", "model.onnx")
io = np.load(os.path.join(W, "refs", f"layer{L}_io.npz")); x = io["in5"]; pad = np.zeros((1, 512, x.shape[1]), np.float32); pad[0, :len(x)] = x
class R(CalibrationDataReader):
    def __init__(self): self.it = iter([{"x": pad}])
    def get_next(self): return next(self.it, None)
out = {}
for name, p in (("fp", fp), ("q", q)):
    aug = p.replace(".onnx", "_aug.onnx"); modify_model_output_intermediate_tensors(p, aug, save_as_external_data=True)
    out[name] = collect_activations(aug, R())
m = create_activation_matching(out["q"], out["fp"])
err = compute_activation_error(m)
rows = sorted(((v.get("xmodel_err", 0), k) for k, v in err.items()), key=lambda t: t[0])
print("lowest SQNR (dB) activations, fp32 vs quantized (higher is better):")
for db, k in rows[:12]: print(f"{db:7.1f}  {k}")
