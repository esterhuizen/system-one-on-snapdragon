"""Does QNN HTP (ort-qnn 1.24.4) run ONNX Gelu(approximate='tanh') natively? fp16 check against numpy."""
import os, collections, math, numpy as np, onnx, onnxruntime as ort
from onnx import TensorProto, helper as oh
x = (np.random.default_rng(0).standard_normal((1, 512, 15360)) * 4).astype(np.float32)
ref = 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
for approx in ("tanh", "none"):
    g = oh.make_graph([oh.make_node("Gelu", ["x"], ["y"], approximate=approx)], "g", [oh.make_tensor_value_info("x", TensorProto.FLOAT, [1, 512, 15360])], [oh.make_tensor_value_info("y", TensorProto.FLOAT, [1, 512, 15360])])
    p = os.path.join(os.environ.get("WINNOW_HOME", "."), "onnx", f"probe_gelu_{approx}.onnx"); onnx.save(oh.make_model(g, opset_imports=[oh.make_opsetid("", 23)], ir_version=10), p)
    so = ort.SessionOptions(); so.log_severity_level = 3; so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    s = ort.InferenceSession(p, so, providers=[("QNNExecutionProvider", {"backend_path": "QnnHtp.dll", "enable_htp_fp16_precision": "1"}), "CPUExecutionProvider"])
    place = {sg.ep_name: dict(collections.Counter(n.op_type for n in sg.get_nodes())) for sg in s.get_provider_graph_assignment_info()}
    print(f"Gelu approximate={approx}: {place}  max|d| vs tanh-GELU {np.abs(s.run(None, {'x': x})[0] - ref).max():.4f}", flush=True)
