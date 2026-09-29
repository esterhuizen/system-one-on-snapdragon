"""Phase 2: run one exported Winnow layer on ORT CPU (fp32), the Hexagon NPU in fp16, and the NPU as w4a16 QDQ
(int4 per-channel MatMul weights, uint16 activations), and compare every output with the PyTorch reference layer.

python phase2_layer.py --layer 0 --mode cpu|fp16|w4a16 [--seq 512] [--n 20]
Test data: refs\\layer<L>_io.npz (6 real sequences' layer inputs/outputs from winnow_ref.py), zero-padded to --seq at
the end (causal mask: padding never influences real positions; only the real rows are compared).
"""
import argparse, collections, os, time
import numpy as np
import onnx
import onnxruntime as ort

W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, required=True); ap.add_argument("--mode", required=True, choices=["cpu", "fp16", "w4a16", "w4blk"])
ap.add_argument("--seq", type=int, default=512); ap.add_argument("--n", type=int, default=20)
ap.add_argument("--weights8", action="store_true", help="w8a16 instead of w4a16 (8-bit MatMul weights)")
ap.add_argument("--block", type=int, default=64, help="w4blk: input-dim block size of the int4 weight scales")
a = ap.parse_args()
L, S = a.layer, a.seq
src = os.path.join(W, "onnx", f"layer{L}_s{S}_fp32", "model.onnx")
io = np.load(os.path.join(W, "refs", f"layer{L}_io.npz"))
n = len([k for k in io.files if k.startswith("in")])
xs, ys, lens = [], [], []
for i in range(n):
    x = io[f"in{i}"]; lens.append(len(x))
    pad = np.zeros((1, S, x.shape[1]), np.float32); pad[0, :len(x)] = x
    xs.append(pad); ys.append(io[f"out{i}"])


def compare(sess, tag):
    errs, rel = [], []
    for x, y, m in zip(xs, ys, lens):
        got = sess.run(["y"], {"x": x})[0][0, :m]
        errs.append(np.abs(got - y).max()); rel.append(np.linalg.norm(got - y) / np.linalg.norm(y))
    print(f"[{tag}] parity vs torch layer: max|d| {max(errs):.4f}  relative L2 error {max(rel):.2e}", flush=True)


def bench(sess, tag):
    feed = {"x": xs[0]}
    for _ in range(3): sess.run(None, feed)
    ts = []
    for _ in range(a.n):
        t = time.perf_counter(); sess.run(None, feed); ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    print(f"[{tag}] latency one layer S={S}: p50 {np.percentile(ts, 50):.1f} ms  min {ts.min():.1f} ms  (x48 layers ~ {np.percentile(ts, 50) * 48 / 1e3:.1f} s)", flush=True)


def qnn(path, fp16):
    so = ort.SessionOptions(); so.log_severity_level = 3
    so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    ctx = path.replace(".onnx", "_ctx.onnx")
    so.add_session_config_entry("ep.context_enable", "1"); so.add_session_config_entry("ep.context_file_path", ctx)
    so.add_session_config_entry("ep.context_embed_mode", "0")
    opts = {"backend_path": "QnnHtp.dll", "htp_performance_mode": "burst", "htp_graph_finalization_optimization_mode": "3"}
    if fp16:
        opts["enable_htp_fp16_precision"] = "1"
    use = ctx if os.path.exists(ctx) else path
    t = time.perf_counter()
    s = ort.InferenceSession(use, so, providers=[("QNNExecutionProvider", opts), "CPUExecutionProvider"])
    print(f"[load] {'cached context' if use == ctx else 'compiled'} in {time.perf_counter() - t:.0f}s", flush=True)
    try:
        for sg in s.get_provider_graph_assignment_info():
            print(f"[placement] {sg.ep_name}: {sum(1 for _ in sg.get_nodes())} nodes {dict(collections.Counter(nd.op_type for nd in sg.get_nodes()))}", flush=True)
    except AttributeError:
        pass
    return s


if a.mode == "cpu":
    s = ort.InferenceSession(src, providers=["CPUExecutionProvider"]); compare(s, "ort-cpu fp32"); bench(s, "ort-cpu fp32")
elif a.mode == "fp16":
    s = qnn(src, True); compare(s, "npu fp16"); bench(s, "npu fp16")
elif a.mode == "w4blk":
    # block-wise int4 weights: rewrite the w8a16 model's per-channel int8 MatMul weights as int4 with one scale per
    # (block of --block input rows, output column) -> DequantizeLinear(axis=0, block_size=B), opset 21+. Activations stay uint16.
    from onnx import numpy_helper as nh, TensorProto, helper as oh
    base = os.path.join(W, "onnx", f"layer{L}_s{S}_w8a16", "model.onnx")
    assert os.path.exists(base), "run --mode w4a16 --weights8 first"
    dst = os.path.join(W, "onnx", f"layer{L}_s{S}_w4blk{a.block}", "model.onnx")
    if not os.path.exists(dst):
        m = onnx.load(base); B = a.block
        init = {i.name: i for i in m.graph.initializer}
        mm_in = {nd.input[1] for nd in m.graph.node if nd.op_type == "MatMul"}
        n_done = 0
        for nd in m.graph.node:
            if nd.op_type != "DequantizeLinear" or nd.output[0] not in mm_in or nd.input[0] not in init:
                continue
            q8 = nh.to_array(init[nd.input[0]]).astype(np.float32); sc = nh.to_array(init[nd.input[1]]).astype(np.float32)
            wf = q8 * sc[None, :]                                           # dequantized weight (K, N)
            K, N = wf.shape; assert K % B == 0
            blk = wf.reshape(K // B, B, N)
            s4 = np.maximum(np.abs(blk).max(1), 1e-12) / 7.0               # symmetric int4 [-8, 7], RTN
            q4 = np.clip(np.round(blk / s4[:, None, :]), -8, 7).astype(np.int8).reshape(K, N)
            for nm in list(nd.input):
                if nm in init: m.graph.initializer.remove(init[nm])
            qn, sn, zn = nd.input[0] + "_q4", nd.input[0] + "_s4", nd.input[0] + "_z4"
            pk = lambda v: ((v[0::2] & 0x0F) | ((v[1::2] & 0x0F) << 4)).astype(np.uint8).tobytes()   # 2 int4 per byte, low nibble first
            m.graph.initializer.extend([oh.make_tensor(qn, TensorProto.INT4, [K, N], pk(q4.flatten()), raw=True),
                                        nh.from_array(s4.astype(np.float32), sn),
                                        oh.make_tensor(zn, TensorProto.INT4, [K // B, N], pk(np.zeros(K // B * N, np.int8)), raw=True)])
            del nd.input[:]; nd.input.extend([qn, sn, zn])
            del nd.attribute[:]; nd.attribute.extend([oh.make_attribute("axis", 0), oh.make_attribute("block_size", B)])
            n_done += 1
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True, location="model.data")
        print(f"[w4blk] {n_done} MatMul weights -> int4, block {B} -> {dst}", flush=True)
    s = qnn(dst, False); compare(s, f"npu w4 block{a.block} a16"); bench(s, f"npu w4 block{a.block} a16")
else:
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config
    wq = QuantType.QInt8 if a.weights8 else QuantType.QInt4
    tag = "w8a16" if a.weights8 else "w4a16"
    dst = os.path.join(W, "onnx", f"layer{L}_s{S}_{tag}", "model.onnx")
    if not os.path.exists(dst):
        class R(CalibrationDataReader):
            def __init__(self): self.it = iter([{"x": x} for x in xs])
            def get_next(self): return next(self.it, None)
        m = onnx.load(src)
        mm = {nd.input[1] for nd in m.graph.node if nd.op_type == "MatMul" and any(i.name == nd.input[1] for i in m.graph.initializer)}
        over = {w: [{"quant_type": wq, "symmetric": True, "axis": 1}] for w in mm}
        t = time.perf_counter()
        cfg = get_qnn_qdq_config(src, R(), activation_type=QuantType.QUInt16, weight_type=QuantType.QUInt16,
                                 per_channel=True, init_overrides=over)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cfg.use_external_data_format = True
        quantize(src, dst, cfg)
        print(f"[quantize] {tag} QDQ model in {time.perf_counter() - t:.0f}s -> {dst}", flush=True)
    s = qnn(dst, False); compare(s, f"npu {tag}"); bench(s, f"npu {tag}")
