"""Change the static sequence length of the packed LPBQ chunks (model_packed.onnx, S=512) to a new S, without rebuilding
or recalibrating: only Reshape target shapes and the graph input/output shapes depend on S (RoPE tables and masks are
already inputs; weights/scales are length-independent). 512 is ambiguous (the global layers' head_dim is also 512), so each
Reshape's role is identified from its neighbours (skipping Q/DQ), as laid out by build_chunk_onnx.py:
  consumer RMSNormalization          (S, H, hd)        -> [0] = S        (q / k / v projections)
  producer Transpose, [2] != NH*hd   (nkv, rep*S, hd)  -> [1] = rep*S    (queries grouped per KV head)
  consumer Add                       (NH, S, S)        -> [1],[2] = S    (scores + mask)
  producer Softmax                   (nkv, rep*S, S)   -> [1] = rep*S, [2] = S
  producer MatMul, consumer Transpose (NH, S, hd)      -> [1] = S
  [0] == 1 and [2] == NH*hd          (1, S, NH*hd)     -> [1] = S        (attention output)
python resize_packed.py --seq 576 [--variant lpbq32]     -> model_packed<S>.onnx next to model_packed.onnx (shares model.data)
"""
import argparse, collections, os
import numpy as np
import onnx
from onnx import numpy_helper as nh

W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); S0, NH = 512, 16
ap = argparse.ArgumentParser(); ap.add_argument("--seq", type=int, required=True); ap.add_argument("--variant", default="lpbq32"); ap.add_argument("--chain", default="chain")
a = ap.parse_args(); S1 = a.seq
for k in range(12):
    d = os.path.join(W, "onnx", a.chain, f"c{k:02d}_{a.variant}")
    m = onnx.load(os.path.join(d, "model_packed.onnx"), load_external_data=False)
    init = {i.name: i for i in m.graph.initializer}
    prod = {o: n for n in m.graph.node for o in n.output}
    cons = collections.defaultdict(list)
    for n in m.graph.node:
        for i in n.input: cons[i].append(n)

    def up(t):                                    # producer op type, skipping Q/DQ
        n = prod.get(t)
        while n is not None and n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            n = prod.get(n.input[0])
        return n.op_type if n is not None else None

    def down(t):                                  # consumer op types, skipping Q/DQ
        out, stack = set(), list(cons.get(t, []))
        while stack:
            n = stack.pop()
            if n.op_type in ("QuantizeLinear", "DequantizeLinear"): stack += cons.get(n.output[0], [])
            else: out.add(n.op_type)
        return out

    roles = collections.Counter()
    for n in m.graph.node:
        if n.op_type != "Reshape": continue
        v = nh.to_array(init[n.input[1]]).copy(); p, c = up(n.input[0]), down(n.output[0])
        if "RMSNormalization" in c:           v[0] = S1; r = "proj"
        elif v[0] == 1 and v[2] == NH * (v[2] // NH) and v[1] == S0 and v[2] in (NH * 256, NH * 512): v[1] = S1; r = "attn_out"
        elif p == "Softmax":                  v[1] = v[1] // S0 * S1; v[2] = S1; r = "probs"
        elif "Add" in c:                      v[1] = S1; v[2] = S1; r = "scores"
        elif p == "Transpose":                v[1] = v[1] // S0 * S1; r = "qgroup"
        elif p == "MatMul" and "Transpose" in c: v[1] = S1; r = "ctx"
        else: raise SystemExit(f"chunk {k}: unrecognised Reshape {n.name} {v.tolist()} producer {p} consumers {c}")
        roles[r] += 1
        name = f"{n.input[1]}__S{S1}__{n.name}"
        m.graph.initializer.append(nh.from_array(v.astype(np.int64), name)); n.input[1] = name
    for vi in list(m.graph.input) + list(m.graph.output):
        dims = vi.type.tensor_type.shape.dim
        for dim in dims:
            if dim.dim_value == S0 and not (vi.name.startswith(("cos_glb", "sin_glb")) and dim is dims[-1]):
                dim.dim_value = S1
    del m.graph.value_info[:]
    onnx.save(m, os.path.join(d, f"model_packed{S1}.onnx"))
    print(f"chunk {k}: reshapes {dict(roles)} -> model_packed{S1}.onnx", flush=True)
