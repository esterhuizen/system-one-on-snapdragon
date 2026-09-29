"""Phase 2: build ONE Winnow-12B (Gemma 4) decoder layer as a static-shape ONNX graph (fp32) from the GGUF weights.

Same maths as winnow_ref.py (llama.cpp gemma4.cpp): RMSNorm as ONNX RMSNormalization (opset 23; the QNN EP places it on the NPU, SimplifiedLayerNormalization falls back to CPU), NeoX RoPE with constant
cos/sin tables, grouped-query attention without repeating K/V (query heads regrouped per KV head), scale 1.0, additive
causal(+window) mask, GELU(tanh) as the ONNX Gelu op, post norms, layer_output_scale.
Input x [1,S,3840] -> output [1,S,3840].      python build_layer_onnx.py --layer 0 --seq 512 --out-dir DIR
"""
import argparse, math, os, sys
import numpy as np
import onnx
from onnx import TensorProto, helper as oh, numpy_helper as nh

W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(W, r"src\winnow-inference\.runtime\llama.cpp\gguf-py"))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, required=True); ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--out-dir", required=True); ap.add_argument("--mask-value", type=float, default=-1e4)
ap.add_argument("--gguf", default=os.path.join(W, r"models\Winnow-12B\gguf\Winnow-12B-Q8_0.gguf"))
a = ap.parse_args()
r = gguf.GGUFReader(a.gguf); T = {t.name: t for t in r.tensors}; F = r.fields
kv = lambda k: F[k].contents()
D = kv("gemma4.embedding_length"); NH = kv("gemma4.attention.head_count"); L = a.layer; S = a.seq
swa = bool(kv("gemma4.attention.sliding_window_pattern")[L]); nkv = kv("gemma4.attention.head_count_kv")[L]
hd = kv("gemma4.attention.key_length_swa") if swa else kv("gemma4.attention.key_length")
base = kv("gemma4.rope.freq_base_swa") if swa else kv("gemma4.rope.freq_base")
EPS = kv("gemma4.attention.layer_norm_rms_epsilon"); WIN = kv("gemma4.attention.sliding_window"); rep = NH // nkv


def w(name):
    t = T[name]
    x = np.asarray(t.data) if t.tensor_type == gguf.GGMLQuantizationType.F32 else dequantize(np.asarray(t.data), t.tensor_type)
    return np.ascontiguousarray(x, dtype=np.float32).reshape([int(s) for s in reversed(t.shape)] if len(t.shape) > 1 else [-1])


p = f"blk.{L}."
inits, nodes = [], []


def const(name, arr):
    inits.append(nh.from_array(np.asarray(arr), name)); return name


def node(op, ins, outs=None, **kw):
    outs = outs or [f"{op}_{len(nodes)}"]
    nodes.append(oh.make_node(op, ins, outs, name=outs[0] + "_n", **kw)); return outs[0]


def rmsnorm(x, weight):
    return node("RMSNormalization", [x, weight], axis=-1, epsilon=EPS)       # opset 23: runs natively on QNN HTP


def matmul(x, name):                      # GGUF (out,in) -> ONNX MatMul weight (in,out)
    return node("MatMul", [x, const(name.replace(".", "_"), w(name).T.copy())])


def shape(*dims):
    return const(f"shape_{len(inits)}", np.array(dims, dtype=np.int64))


# RoPE tables (NeoX): out = x*cos + rotate_half(x)*sin, rotate_half(x) = [-x2, x1]
half = hd // 2
inv = base ** (-np.arange(half, dtype=np.float64) * 2 / hd)
if not swa:
    inv = inv / w("rope_freqs.weight").astype(np.float64)
ang = np.arange(S, dtype=np.float64)[:, None] * inv[None, :]
cos = const("rope_cos", np.concatenate([np.cos(ang)] * 2, -1).astype(np.float32)[:, None, :])     # [S,1,hd]
sin = const("rope_sin", np.concatenate([np.sin(ang)] * 2, -1).astype(np.float32)[:, None, :])
qpos, kpos = np.arange(S)[:, None], np.arange(S)[None, :]
masked = (kpos > qpos) | ((qpos - kpos >= WIN) if swa else False)
mask = const("mask", np.where(masked, a.mask_value, 0.0).astype(np.float32)[None])                  # [1,S,S]
sl = lambda x, s, e: node("Slice", [x, const(f"s{len(inits)}", np.array([s])), const(f"e{len(inits)}", np.array([e])), const(f"a{len(inits)}", np.array([-1]))])


def rope(x):
    x1, x2 = sl(x, 0, half), sl(x, half, hd)
    rot = node("Concat", [node("Neg", [x2]), x1], axis=-1)
    return node("Add", [node("Mul", [x, cos]), node("Mul", [rot, sin])])


x = "x"
h = rmsnorm(x, const("attn_norm", w(p + "attn_norm.weight")))
q = node("Reshape", [matmul(h, p + "attn_q.weight"), shape(S, NH, hd)])
kraw = node("Reshape", [matmul(h, p + "attn_k.weight"), shape(S, nkv, hd)])
v = node("Reshape", [matmul(h, p + "attn_v.weight"), shape(S, nkv, hd)]) if p + "attn_v.weight" in T else kraw
q = rope(rmsnorm(q, const("q_norm", w(p + "attn_q_norm.weight"))))
k = rope(rmsnorm(kraw, const("k_norm", w(p + "attn_k_norm.weight"))))
v = rmsnorm(v, const("v_ones", np.ones(hd, np.float32)))
qg = node("Reshape", [node("Transpose", [q], perm=[1, 0, 2]), shape(nkv, rep * S, hd)])            # heads grouped per KV head
kt = node("Transpose", [k], perm=[1, 2, 0])                                                         # [nkv,hd,S]
sc = node("Reshape", [node("MatMul", [qg, kt]), shape(NH, S, S)])                                  # scale 1.0
pr = node("Reshape", [node("Softmax", [node("Add", [sc, mask])], axis=-1), shape(nkv, rep * S, S)])
o = node("MatMul", [pr, node("Transpose", [v], perm=[1, 0, 2])])                                     # [nkv,rep*S,hd]
o = node("Reshape", [node("Transpose", [node("Reshape", [o, shape(NH, S, hd)])], perm=[1, 0, 2]), shape(1, S, NH * hd)])
attn_out = node("Add", [rmsnorm(matmul(o, p + "attn_output.weight"), const("post_attn_norm", w(p + "post_attention_norm.weight"))), x])
f = rmsnorm(attn_out, const("ffn_norm", w(p + "ffn_norm.weight")))
g = matmul(f, p + "ffn_gate.weight")
gelu = node("Gelu", [g], approximate="tanh")      # native on QNN HTP; the written-out tanh formula quantized badly
f = matmul(node("Mul", [gelu, matmul(f, p + "ffn_up.weight")]), p + "ffn_down.weight")
y = node("Add", [rmsnorm(f, const("post_ffw_norm", w(p + "post_ffw_norm.weight"))), attn_out])
node("Mul", [y, const("out_scale", w(p + "layer_output_scale.weight").reshape(()))], ["y"])

graph = oh.make_graph(nodes, f"winnow_layer{L}_s{S}", [oh.make_tensor_value_info("x", TensorProto.FLOAT, [1, S, D])],
                      [oh.make_tensor_value_info("y", TensorProto.FLOAT, [1, S, D])], inits)
model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 23)], producer_name="winnow-qnn-phase2", ir_version=10)
os.makedirs(a.out_dir, exist_ok=True)
out = os.path.join(a.out_dir, "model.onnx")
onnx.save(model, out, save_as_external_data=True, all_tensors_to_one_file=True, location="model.data", size_threshold=1024)
print(f"layer {L} ({'swa' if swa else 'global'}, hd {hd}, kv heads {nkv}) S={S}: {len(nodes)} nodes -> {out}")
