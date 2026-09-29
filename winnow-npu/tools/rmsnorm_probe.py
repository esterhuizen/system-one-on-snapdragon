"""Which RMSNorm formulation does the QNN HTP EP (onnxruntime-qnn 1.24.4) take? Small [512,3840] graphs, fp16 HTP."""
import os, collections, numpy as np, onnx, onnxruntime as ort
from onnx import TensorProto, helper as oh, numpy_helper as nh
S, D, EPS = 512, 3840, 1e-6
w = np.random.default_rng(0).uniform(0.5, 1.5, D).astype(np.float32)
x = (np.random.default_rng(1).standard_normal((1, S, D)) * 3).astype(np.float32)
ref = x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w
def variant(name):
    I = [nh.from_array(w, "w")]; opset = 17
    if name == "sln_nostash":
        N = [oh.make_node("SimplifiedLayerNormalization", ["x", "w"], ["y"], axis=-1, epsilon=EPS)]
    elif name == "rmsnorm23":
        N = [oh.make_node("RMSNormalization", ["x", "w"], ["y"], axis=-1, epsilon=EPS)]; opset = 23
    else:
        I += [nh.from_array(np.array(EPS, np.float32), "eps"), nh.from_array(np.array([-1], np.int64), "ax")]
        N = [oh.make_node("Mul", ["x", "x"], ["sq"]), oh.make_node("ReduceMean", ["sq", "ax"], ["ms"], keepdims=1),
             oh.make_node("Add", ["ms", "eps"], ["mse"]), oh.make_node("Sqrt", ["mse"], ["rt"]),
             oh.make_node("Div", ["x", "rt"], ["xn"]), oh.make_node("Mul", ["xn", "w"], ["y"])]; opset = 18
    g = oh.make_graph(N, name, [oh.make_tensor_value_info("x", TensorProto.FLOAT, [1, S, D])], [oh.make_tensor_value_info("y", TensorProto.FLOAT, [1, S, D])], I)
    return oh.make_model(g, opset_imports=[oh.make_opsetid("", opset)], ir_version=10)
for v in ("sln_nostash", "rmsnorm23", "decomposed"):
    p = os.path.join(os.environ.get("WINNOW_HOME", "."), "onnx", f"probe_{v}.onnx"); onnx.save(variant(v), p)
    so = ort.SessionOptions(); so.log_severity_level = 3; so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    try:
        s = ort.InferenceSession(p, so, providers=[("QNNExecutionProvider", {"backend_path": "QnnHtp.dll", "enable_htp_fp16_precision": "1"}), "CPUExecutionProvider"])
        place = {sg.ep_name: dict(collections.Counter(n.op_type for n in sg.get_nodes())) for sg in s.get_provider_graph_assignment_info()}
        y = s.run(None, {"x": x})[0]
        print(f"{v:12s} placement {place}  max|d| {np.abs(y - ref).max():.4f}", flush=True)
    except Exception as e:
        print(f"{v:12s} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
