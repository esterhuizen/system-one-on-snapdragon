"""Phase 1: PyTorch reference of Winnow-12B (Gemma 4) built directly from the GGUF, checked against the CPU server.

Streams one decoder layer at a time (Q8_0 -> fp32), so it needs only a few GB. Follows llama.cpp src/models/gemma4.cpp:
sqrt(d) embedding scale; RMSNorm (weights as stored); q/k RMSNorm per head, v RMSNorm without weight; V = K projection on
layers without attn_v; NeoX RoPE (global layers: base 1e6, rope_freqs as frequency divisors; SWA layers: base 1e4);
attention scale 1.0; causal (+1024-token window on SWA layers); GELU(tanh)-gated MLP; post-attn / post-ffn norms;
layer_output_scale; output norm; answer logits = h . token_embd[label] soft-capped at 30 (Winnow's selected head).

python winnow_ref.py --ref refs\\ref.json --out refs\\torch_ref.json [--save-layers 0,5] [--gguf ...]
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(W, r"src\winnow-inference\.runtime\llama.cpp\gguf-py"))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--gguf", default=os.path.join(W, r"models\Winnow-12B\gguf\Winnow-12B-Q8_0.gguf"))
ap.add_argument("--save-layers", default="0,5", help="layers whose input/output hidden states are saved for Phase 2")
ap.add_argument("--threads", type=int, default=12)
a = ap.parse_args()
torch.set_num_threads(a.threads)

r = gguf.GGUFReader(a.gguf)
T = {t.name: t for t in r.tensors}
F = {k: f for k, f in r.fields.items()}


def kv(name):
    return F[name].contents()


D = kv("gemma4.embedding_length"); NL = kv("gemma4.block_count"); NH = kv("gemma4.attention.head_count")
NKV = kv("gemma4.attention.head_count_kv"); SWA = kv("gemma4.attention.sliding_window_pattern")
HD, HD_SWA = kv("gemma4.attention.key_length"), kv("gemma4.attention.key_length_swa")
BASE, BASE_SWA = kv("gemma4.rope.freq_base"), kv("gemma4.rope.freq_base_swa")
EPS = kv("gemma4.attention.layer_norm_rms_epsilon"); WIN = kv("gemma4.attention.sliding_window")
CAP = kv("gemma4.final_logit_softcapping")


def W_(name):
    t = T[name]
    x = dequantize(np.asarray(t.data), t.tensor_type) if t.tensor_type not in (gguf.GGMLQuantizationType.F32,) else np.asarray(t.data)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape([int(s) for s in reversed(t.shape)] if len(t.shape) > 1 else [-1])


def embed_rows(ids):
    """Dequantize only the needed rows of token_embd (Q8_0: 34-byte blocks of 32)."""
    t = T["token_embd.weight"]; raw = np.asarray(t.data)
    rows = raw.reshape(int(t.shape[1]), -1)[np.asarray(ids)]
    return torch.from_numpy(np.ascontiguousarray(dequantize(rows, t.tensor_type), dtype=np.float32)).reshape(len(ids), D)


def rms(x, w=None):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return y * w if w is not None else y


def rope(x, pos, base, n_rot, freq_factors=None):
    """NeoX RoPE over the first n_rot dims of the head. x: (S, H, hd)."""
    half = n_rot // 2
    inv = base ** (-torch.arange(0, half, dtype=torch.float64) * 2 / n_rot)
    if freq_factors is not None:
        inv = inv / freq_factors.double()
    ang = pos[:, None].double() * inv[None, :]
    cos, sin = ang.cos().float()[:, None, :], ang.sin().float()[:, None, :]
    x1, x2 = x[..., :half], x[..., half:n_rot]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, x[..., n_rot:]], -1)


def gelu_tanh(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))


refs = json.load(open(a.ref, encoding="utf-8"))
labels = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
tokens = kv("tokenizer.ggml.tokens")
tok_index = {s: i for i, s in enumerate(tokens)}
label_ids = [tok_index[l] for l in labels]
seqs = []   # one sequence per (request, question): prefix ids + suffix ids, as Winnow scores it
for R in refs:
    pre = R["inspect"]["prefix_token_ids"]
    for qi, (qid, ans) in enumerate(R["answers"].items()):
        suf = R["inspect"]["suffix_token_ids"][qi]
        seqs.append(dict(req=R["id"], q=qid, ids=pre + suf, n=len(ans["winnow"]["logits"]), ref=ans["winnow"]["logits"]))
print(f"{len(seqs)} sequences, lengths {[len(s['ids']) for s in seqs]}; label ids {label_ids[:4]}...", flush=True)

H = [embed_rows(s["ids"]) * math.sqrt(D) for s in seqs]
save = {int(x) for x in a.save_layers.split(",") if x != ""}
dump = {}
rope_freqs = W_("rope_freqs.weight")
t0 = time.perf_counter()
for L in range(NL):
    p = f"blk.{L}."
    swa = bool(SWA[L]); hd = HD_SWA if swa else HD; nkv = NKV[L]
    w = {k: W_(p + k + ".weight") for k in ("attn_norm", "attn_q", "attn_k", "attn_q_norm", "attn_k_norm", "attn_output",
                                            "post_attention_norm", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down", "post_ffw_norm",
                                            "layer_output_scale")}
    wv = W_(p + "attn_v.weight") if p + "attn_v.weight" in T else None
    for i, s in enumerate(seqs):
        x = H[i]; S = x.shape[0]; pos = torch.arange(S)
        if L in save:
            dump.setdefault(L, {})[i] = {"in": x.clone()}
        h = rms(x, w["attn_norm"])
        q = (h @ w["attn_q"].T).view(S, NH, hd)
        k = (h @ w["attn_k"].T).view(S, nkv, hd)
        v = (h @ wv.T).view(S, nkv, hd) if wv is not None else k
        q = rms(q, w["attn_q_norm"]); v = rms(v); k = rms(k, w["attn_k_norm"])
        ff = None if swa else rope_freqs
        q = rope(q, pos, BASE_SWA if swa else BASE, hd, ff); k = rope(k, pos, BASE_SWA if swa else BASE, hd, ff)
        rep = NH // nkv
        k = k.repeat_interleave(rep, 1); v = v.repeat_interleave(rep, 1)
        att = torch.einsum("qhd,khd->hqk", q, k)          # scale 1.0
        mask = pos[None, :] > pos[:, None]
        if swa:
            mask = mask | (pos[:, None] - pos[None, :] >= WIN)
        att = att.masked_fill(mask[None], float("-inf")).softmax(-1)
        o = torch.einsum("hqk,khd->qhd", att, v).reshape(S, NH * hd) @ w["attn_output"].T
        attn_out = rms(o, w["post_attention_norm"]) + x
        f = rms(attn_out, w["ffn_norm"])
        f = (gelu_tanh(f @ w["ffn_gate"].T) * (f @ w["ffn_up"].T)) @ w["ffn_down"].T
        x = (rms(f, w["post_ffw_norm"]) + attn_out) * w["layer_output_scale"]
        H[i] = x
        if L in save:
            dump[L][i]["out"] = x.clone()
    del w, wv
    print(f"layer {L:2d} ({'swa' if swa else 'global'}) done  {time.perf_counter() - t0:.0f}s", flush=True)

E = embed_rows(label_ids)
onorm = W_("output_norm.weight")
res = []
for i, s in enumerate(seqs):
    h = rms(H[i][-1], onorm)
    lg = (E[: s["n"]] @ h)
    lg = CAP * torch.tanh(lg / CAP)
    ref = torch.tensor(s["ref"])
    p, pr = (lg).softmax(0), ref.softmax(0)
    res.append(dict(req=s["req"], q=s["q"], ours=lg.tolist(), ref=s["ref"], max_abs_logit_diff=float((lg - ref).abs().max()),
                    max_abs_prob_diff=float((p - pr).abs().max()), same_argmax=bool(lg.argmax() == ref.argmax())))
    print(f"{s['req']:28s} {s['q']:10s} max|dlogit| {res[-1]['max_abs_logit_diff']:.3f}  max|dp| {res[-1]['max_abs_prob_diff']:.4f}  "
          f"argmax same {res[-1]['same_argmax']}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
for L, d in dump.items():
    torch.save({i: v for i, v in d.items()}, os.path.join(os.path.dirname(a.out), f"layer{L}_io.pt"))
print("saved", a.out)
