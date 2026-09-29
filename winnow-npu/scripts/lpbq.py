"""Convert a w8a16 QDQ model's MatMul weights to Qualcomm LPBQ int4 (low-power block quantization) in the exact pattern
onnxruntime's QNN EP (v1.24.4 qnn_node_group/lpbqmatmul_fusion.cc) fuses into a native HTP MatMul:

    u8 block scales (K/B, N) --DequantizeLinear(scale = per-channel float (N,), zp 0, axis=1)--> block scales (K/B, N)
    int4 weight (K, N) --DequantizeLinear(scale = ^, zp int4 0, axis=0, block_size=B)--> MatMul input 1

Each block's ideal scale s[b,n] is chosen by an MSE search over clipping ratios (not just max/7); pc[n] = max_b s / 15,
u8 = round(s / pc) in 1..15 (QNN reads LPBQ int4 block scales as 4-bit), q4 = clip(round(w / (pc*u8)), -8, 7). Activations keep their uint16 QDQ.

python lpbq.py --src w8a16/model.onnx --dst lpbq/model.onnx [--block 64] [--no-mse]
Also importable: convert(src, dst, block, mse).
"""
import argparse, os, time
import numpy as np
import onnx
from onnx import TensorProto, helper as oh, numpy_helper as nh

BS_MAX = 15
RATIOS = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7], np.float32)


def pack4(v):
    v = v.astype(np.int8).ravel()
    return ((v[0::2] & 0x0F) | ((v[1::2] & 0x0F) << 4)).astype(np.uint8).tobytes()


def quant_blocks(wf, B, mse):
    K, N = wf.shape
    blk = wf.reshape(K // B, B, N)
    amax = np.maximum(np.abs(blk).max(1), 1e-12)                        # (K/B, N)
    if mse:
        best, best_err = amax / 7.0, None
        for r in RATIOS:
            s = amax * r / 7.0
            q = np.clip(np.round(blk / s[:, None, :]), -8, 7)
            err = ((q * s[:, None, :] - blk) ** 2).sum(1)
            if best_err is None:
                best_err = err
            else:
                better = err < best_err; best = np.where(better, s, best); best_err = np.where(better, err, best_err)
        s_ideal = best
    else:
        s_ideal = amax / 7.0
    # QNN LPBQ for int4 weights stores block scales with blockScaleBitwidth=4 (ORT qnn_quant_params_wrapper.cc):
    # integers 1..15 times the per-channel float scale (8-bit values are misread -> garbage on the NPU)
    pc = s_ideal.max(0) / BS_MAX                                         # (N,)
    u8 = np.clip(np.round(s_ideal / pc[None, :]), 1, BS_MAX).astype(np.uint8)
    s = u8.astype(np.float32) * pc[None, :]
    q4 = np.clip(np.round(blk / s[:, None, :]), -8, 7).astype(np.int8).reshape(K, N)
    return q4, u8, pc.astype(np.float32)


def convert(src, dst, B=64, mse=True, only=None):
    """only: substrings; convert just the MatMul weights whose name contains one of them (e.g. ['ffn_'])"""
    m = onnx.load(src)
    init = {i.name: i for i in m.graph.initializer}
    mm_in = {nd.input[1] for nd in m.graph.node if nd.op_type == "MatMul"}
    new_nodes, n = [], 0
    for nd in m.graph.node:
        if nd.op_type == "DequantizeLinear" and nd.output[0] in mm_in and nd.input[0] in init and (not only or any(o in nd.input[0] for o in only)):
            q8 = nh.to_array(init[nd.input[0]]).astype(np.float32); sc = nh.to_array(init[nd.input[1]]).astype(np.float32)
            wf = q8 * (sc[None, :] if sc.ndim == 1 else sc)
            K, N = wf.shape; assert K % B == 0, (nd.input[0], K, B)
            q4, u8, pc = quant_blocks(wf, B, mse)
            for nm in list(nd.input):
                if nm in init: m.graph.initializer.remove(init.pop(nm))
            b = nd.input[0]
            names = dict(q=b + "_lpbq_q4", zq=b + "_lpbq_zq4", u8=b + "_lpbq_u8", pc=b + "_lpbq_pc", zu8=b + "_lpbq_zu8", s=b + "_lpbq_scale")
            m.graph.initializer.extend([
                oh.make_tensor(names["q"], TensorProto.INT4, [K, N], pack4(q4), raw=True),
                oh.make_tensor(names["zq"], TensorProto.INT4, [K // B, N], pack4(np.zeros(K // B * N, np.int8)), raw=True),
                nh.from_array(u8, names["u8"]), nh.from_array(pc, names["pc"]), nh.from_array(np.zeros(N, np.uint8), names["zu8"])])
            new_nodes.append(oh.make_node("DequantizeLinear", [names["u8"], names["pc"], names["zu8"]], [names["s"]],
                                          name=names["s"] + "_dql", axis=1))
            del nd.input[:]; nd.input.extend([names["q"], names["s"], names["zq"]])
            del nd.attribute[:]; nd.attribute.extend([oh.make_attribute("axis", 0), oh.make_attribute("block_size", B)])
            n += 1
        new_nodes.append(nd)
    del m.graph.node[:]; m.graph.node.extend(new_nodes)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True, location="model.data")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--block", type=int, default=64); ap.add_argument("--no-mse", action="store_true")
    ap.add_argument("--only", default="", help="comma list of weight-name substrings to convert (default all)")
    a = ap.parse_args(); t = time.perf_counter()
    k = convert(a.src, a.dst, a.block, not a.no_mse, [s for s in a.only.split(",") if s] or None)
    print(f"[lpbq] {k} MatMul weights -> int4 LPBQ block {a.block}{' (MSE clip)' if not a.no_mse else ''} in {time.perf_counter() - t:.0f}s -> {a.dst}")
