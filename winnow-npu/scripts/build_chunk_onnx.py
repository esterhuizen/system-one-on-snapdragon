"""Phase 3: build a CHUNK of consecutive Winnow-12B (Gemma 4) decoder layers [start, end) as one static-shape fp32 ONNX graph.

Per-layer maths identical to build_layer_onnx.py (validated in Phase 2: fp32 ORT == torch to 1e-6): RMSNormalization and
Gelu(tanh) (native on QNN HTP, opset 23), NeoX RoPE tables, grouped-query attention, scale 1.0, additive causal(+window)
mask (-60), post norms, layer_output_scale. RoPE tables and masks are shared by all layers of the same kind in the chunk.
Input x [1,S,3840] -> output y [1,S,3840].   python build_chunk_onnx.py --start 0 --end 4 --seq 512 --out-dir DIR
"""
import argparse, os, sys
import numpy as np
import onnx
from onnx import TensorProto, helper as oh, numpy_helper as nh

W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(W, r"src\winnow-inference\.runtime\llama.cpp\gguf-py"))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, required=True); ap.add_argument("--end", type=int, required=True)
ap.add_argument("--seq", type=int, default=512); ap.add_argument("--out-dir", required=True)
ap.add_argument("--mask-value", type=float, default=-60.0)
ap.add_argument("--gguf", default=os.path.join(W, r"models\Winnow-12B\gguf\Winnow-12B-Q8_0.gguf"))
a = ap.parse_args()
r = gguf.GGUFReader(a.gguf); T = {t.name: t for t in r.tensors}; F = r.fields
kv = lambda k: F[k].contents()
D = kv("gemma4.embedding_length"); NH = kv("gemma4.attention.head_count"); S = a.seq
PAT = kv("gemma4.attention.sliding_window_pattern"); NKV = kv("gemma4.attention.head_count_kv")
EPS = kv("gemma4.attention.layer_norm_rms_epsilon"); WIN = kv("gemma4.attention.sliding_window")


def w(name):
    t = T[name]
    x = np.asarray(t.data) if t.tensor_type == gguf.GGMLQuantizationType.F32 else dequantize(np.asarray(t.data), t.tensor_type)
    return np.ascontiguousarray(x, dtype=np.float32).reshape([int(s) for s in reversed(t.shape)] if len(t.shape) > 1 else [-1])


inits, nodes, made = [], [], set()


def const(name, arr):
    if name not in made:
        inits.append(nh.from_array(np.asarray(arr), name)); made.add(name)
    return name


def node(op, ins, outs=None, **kw):
    outs = outs or [f"{op}_{len(nodes)}"]
    nodes.append(oh.make_node(op, ins, outs, name=outs[0] + "_n", **kw)); return outs[0]


shape = lambda *dims: const("shape_" + "_".join(map(str, dims)), np.array(dims, dtype=np.int64))
i64 = lambda v: const(f"i64_{v}", np.array([v], dtype=np.int64))
rmsnorm = lambda x, wt: node("RMSNormalization", [x, wt], axis=-1, epsilon=EPS)


def tables(swa):
    """RoPE cos/sin [S,1,hd] and additive mask [1,S,S] for one layer kind (shared across the chunk)."""
    k = "swa" if swa else "glb"
    hd = kv("gemma4.attention.key_length_swa") if swa else kv("gemma4.attention.key_length")
    if f"cos_{k}" not in made:
        base = kv("gemma4.rope.freq_base_swa") if swa else kv("gemma4.rope.freq_base")
        inv = base ** (-np.arange(hd // 2, dtype=np.float64) * 2 / hd)
        if not swa:
            inv = inv / w("rope_freqs.weight").astype(np.float64)
        ang = np.arange(S, dtype=np.float64)[:, None] * inv[None, :]
        const(f"cos_{k}", np.concatenate([np.cos(ang)] * 2, -1).astype(np.float32)[:, None, :])
        const(f"sin_{k}", np.concatenate([np.sin(ang)] * 2, -1).astype(np.float32)[:, None, :])
        qp, kp = np.arange(S)[:, None], np.arange(S)[None, :]
        m = (kp > qp) | ((qp - kp >= WIN) if swa else False)
        const(f"mask_{k}", np.where(m, a.mask_value, 0.0).astype(np.float32)[None])
    return f"cos_{k}", f"sin_{k}", f"mask_{k}", hd


def layer(L, x):
    p = f"blk.{L}."; swa = bool(PAT[L]); nkv = NKV[L]; rep = NH // nkv
    cos, sin, mask, hd = tables(swa); half = hd // 2
    mm = lambda t, name: node("MatMul", [t, const(f"l{L}_" + name, w(p + name + ".weight").T.copy())])
    nw = lambda name: const(f"l{L}_" + name, w(p + name + ".weight"))

    def rope(t):
        x1 = node("Slice", [t, i64(0), i64(half), i64(-1)]); x2 = node("Slice", [t, i64(half), i64(hd), i64(-1)])
        rot = node("Concat", [node("Neg", [x2]), x1], axis=-1)
        return node("Add", [node("Mul", [t, cos]), node("Mul", [rot, sin])])

    h = rmsnorm(x, nw("attn_norm"))
    q = node("Reshape", [mm(h, "attn_q"), shape(S, NH, hd)])
    kraw = node("Reshape", [mm(h, "attn_k"), shape(S, nkv, hd)])
    v = node("Reshape", [mm(h, "attn_v"), shape(S, nkv, hd)]) if p + "attn_v.weight" in T else kraw
    q = rope(rmsnorm(q, nw("attn_q_norm"))); k = rope(rmsnorm(kraw, nw("attn_k_norm")))
    v = rmsnorm(v, const(f"ones_{hd}", np.ones(hd, np.float32)))
    qg = node("Reshape", [node("Transpose", [q], perm=[1, 0, 2]), shape(nkv, rep * S, hd)])
    sc = node("Reshape", [node("MatMul", [qg, node("Transpose", [k], perm=[1, 2, 0])]), shape(NH, S, S)])
    pr = node("Reshape", [node("Softmax", [node("Add", [sc, mask])], axis=-1), shape(nkv, rep * S, S)])
    o = node("MatMul", [pr, node("Transpose", [v], perm=[1, 0, 2])])
    o = node("Reshape", [node("Transpose", [node("Reshape", [o, shape(NH, S, hd)])], perm=[1, 0, 2]), shape(1, S, NH * hd)])
    attn_out = node("Add", [rmsnorm(mm(o, "attn_output"), nw("post_attention_norm")), x])
    f = rmsnorm(attn_out, nw("ffn_norm"))
    f = mm(node("Mul", [node("Gelu", [mm(f, "ffn_gate")], approximate="tanh"), mm(f, "ffn_up")]), "ffn_down")
    y = node("Add", [rmsnorm(f, nw("post_ffw_norm")), attn_out])
    return node("Mul", [y, const(f"l{L}_out_scale", w(p + "layer_output_scale.weight").reshape(()))])


x = "x"
for L in range(a.start, a.end):
    x = layer(L, x)
nodes[-1].output[0] = "y"; nodes[-1].name = "y_n"
graph = oh.make_graph(nodes, f"winnow_l{a.start}_{a.end}_s{S}", [oh.make_tensor_value_info("x", TensorProto.FLOAT, [1, S, D])],
                      [oh.make_tensor_value_info("y", TensorProto.FLOAT, [1, S, D])], inits)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 23)], producer_name="winnow-qnn-phase3", ir_version=10)
os.makedirs(a.out_dir, exist_ok=True)
out = os.path.join(a.out_dir, "model.onnx")
onnx.save(model, out, save_as_external_data=True, all_tensors_to_one_file=True, location="model.data", size_threshold=1024)
print(f"layers {a.start}-{a.end - 1} S={S}: {len(nodes)} nodes -> {out}", flush=True)
