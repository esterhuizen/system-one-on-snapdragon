"""Winnow-12B on the Snapdragon Hexagon NPU: a Jev-compatible decision runtime (no CPU Winnow server needed).

Request (Jev / TypeSafe wire format) -> winnow_prompt.encode (token-exact port of Winnow's rendering) -> questions packed
greedily into passes of at most S tokens (prefix once + question blocks; block attention mask; positions restart at the
prefix end, so each question sees exactly what Winnow's per-question branch sees) -> 12 compiled LPBQ-int4 QNN chunks
(4 layers each, S-token static graphs) -> CPU head: RMSNorm . token_embd[label rows], soft-cap 30, softmax over options.

    rt = WinnowNPU(chain="chain_pub", seq=576); rt.decide({"state": ..., "questions": {...}}) -> {"answers": ..., ...}
Needs onnxruntime>=1.30 + onnxruntime-qnn plugin (LPBQ), the compiled contexts (compile_chain.py) and the
Q8_0 GGUF (embeddings, output norm, answer-label rows).
"""
import json, math, os, sys, threading, time
import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_qnn as oq

W = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(W, "src", "winnow-inference", ".runtime", "llama.cpp", "gguf-py"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402
import winnow_prompt as wp  # noqa: E402

D, CAP, MASK = 3840, 30.0, -60.0


class WinnowNPU:
    def __init__(self, chain="chain_pub", seq=576, variant="lpbq32", gguf_path=None, perf="burst"):
        self.S = seq; mp = "model_packed" if seq == 512 else f"model_packed{seq}"
        g = gguf.GGUFReader(gguf_path or os.path.join(W, "models", "Winnow-12B", "gguf", "Winnow-12B-Q8_0.gguf"))
        T = {t.name: t for t in g.tensors}; kv = lambda k: g.fields[k].contents()
        E = T["token_embd.weight"]; self._raw, self._etype = np.asarray(E.data).reshape(int(E.shape[1]), -1), E.tensor_type
        self.onorm = np.asarray(T["output_norm.weight"].data, np.float32); self.eps = kv("gemma4.attention.layer_norm_rms_epsilon")
        self.lab = self.emb([i for _, i in wp.label_list()])
        rf = np.asarray(T["rope_freqs.weight"].data, np.float64)
        self.inv = {"swa": kv("gemma4.rope.freq_base_swa") ** (-np.arange(128, dtype=np.float64) * 2 / 256),
                    "glb": kv("gemma4.rope.freq_base") ** (-np.arange(256, dtype=np.float64) * 2 / 512) / rf}
        if not any(x.ep_name == oq.EP_NAME for x in ort.get_ep_devices()):
            ort.register_execution_provider_library(oq.EP_NAME, oq.get_library_path())
        npu = [x for x in ort.get_ep_devices() if x.ep_name == oq.EP_NAME and x.device.type == ort.OrtHardwareDeviceType.NPU]
        if not npu:
            raise RuntimeError("no QNN NPU device")
        self.sess = []
        for k in range(12):
            d = os.path.join(W, "onnx", chain, f"c{k:02d}_{variant}"); ctx = os.path.join(d, mp + "_ctx.onnx")
            if not os.path.exists(ctx):
                raise FileNotFoundError(f"{ctx} missing: compile first (compile_chain.py)")
            so = ort.SessionOptions(); so.log_severity_level = 3
            so.add_provider_for_devices(npu, {"htp_performance_mode": perf})
            s = ort.InferenceSession(ctx, so)
            m = onnx.load(os.path.join(d, mp + ".onnx"), load_external_data=False); I = {i.name: i for i in m.graph.initializer}
            qp = {}
            for inp in s.get_inputs():
                if inp.name != "x":
                    b = inp.name[: -len("_quantized")]
                    qp[inp.name] = (b, float(onnx.numpy_helper.to_array(I[b + "_scale"])), int(onnx.numpy_helper.to_array(I[b + "_zero_point"])))
            self.sess.append((s, qp))
        self.lock = threading.Lock()

    def emb(self, ids):
        return dequantize(self._raw[np.asarray(ids)], self._etype).reshape(len(ids), D).astype(np.float32)

    def _tables(self, prefix_len, q_lens):
        S = self.S; pos = np.zeros(S, np.float64); grp = np.full(S, -1)
        pos[:prefix_len] = np.arange(prefix_len); grp[:prefix_len] = 0; o = prefix_len
        for qi, L in enumerate(q_lens, 1):
            pos[o: o + L] = prefix_len + np.arange(L); grp[o: o + L] = qi; o += L
        pos[o:] = np.arange(S - o)
        i, j = np.arange(S)[:, None], np.arange(S)[None, :]
        ok = ((j <= i) & ((grp[None, :] == 0) | (grp[None, :] == grp[:, None]))) | (i == j)
        mask = np.where(ok, 0.0, MASK).astype(np.float32)[None]
        t = {"mask_swa": mask, "mask_glb": mask}
        for kind in ("swa", "glb"):
            ang = pos[:, None] * self.inv[kind][None, :]
            t[f"cos_{kind}"] = np.concatenate([np.cos(ang)] * 2, -1).astype(np.float32)[:, None, :]
            t[f"sin_{kind}"] = np.concatenate([np.sin(ang)] * 2, -1).astype(np.float32)[:, None, :]
        return t

    def _pass(self, prefix, suffixes):
        """One NPU pass over prefix + the given question blocks -> answer-slot logits (one row per block)."""
        ids = prefix + [t for s in suffixes for t in s]
        x = np.zeros((1, self.S, D), np.float32); x[0, : len(ids)] = self.emb(ids) * math.sqrt(D)
        tab = self._tables(len(prefix), [len(s) for s in suffixes])
        dbg = os.environ.get("WINNOW_NPU_DEBUG") == "1"
        for k, (s, qp) in enumerate(self.sess):
            feed = {"x": x}
            for name, (base, sc, zp) in qp.items():
                feed[name] = np.clip(np.round(tab[base] / sc) + zp, 0, 65535).astype(np.uint16)
            x = s.run(["y"], feed)[0]
            if dbg:
                live = x[0, : len(ids)]
                print(f"[dbg] chunk {k:2d} finite {bool(np.isfinite(x).all())} max|y| {float(np.abs(live).max()):9.2f} "
                      f"mean|y| {float(np.abs(live).mean()):7.3f}", flush=True)
        rows, end = [], len(prefix)
        for suf in suffixes:
            end += len(suf); h = x[0, end - 1]; rows.append(h / np.sqrt((h * h).mean() + self.eps) * self.onorm)
        return rows

    def decide(self, body):
        e = wp.encode(body); pre, sufs, qs = e["prefix_ids"], e["suffix_ids"], e["questions"]
        for s, q in zip(sufs, qs):
            if len(pre) + len(s) > self.S:
                raise ValueError(f"question {q['id']!r}: state + question is {len(pre) + len(s)} tokens > {self.S}")
        packs, cur, used = [], [], len(pre)                      # greedy packing into passes of <= S tokens
        for i, s in enumerate(sufs):
            if cur and used + len(s) > self.S:
                packs.append(cur); cur, used = [], len(pre)
            cur.append(i); used += len(s)
        packs.append(cur)
        answers, t0 = {}, time.perf_counter()
        with self.lock:
            for pk in packs:
                for i, h in zip(pk, self._pass(pre, [sufs[i] for i in pk])):
                    q = qs[i]; n = len(q["keys"])
                    lg = CAP * np.tanh((self.lab[:n] @ h) / CAP); p = np.exp(lg - lg.max()); p /= p.sum()
                    probs = {k: float(v) for k, v in zip(q["keys"], p)}
                    ent = -sum(float(v) * math.log(float(v)) for v in p if v > 0); conf = float(max(0.0, min(1.0, 1 - ent / math.log(n))))
                    if q["type"] == "noul":
                        answers[q["id"]] = {"type": "noul", "noul": float(p[1])}
                    elif q["type"] == "choice":
                        answers[q["id"]] = {"type": "choice", "choice": q["keys"][int(p.argmax())], "probabilities": probs, "confidence": conf}
                    else:
                        answers[q["id"]] = {"type": "score", "score": float(sum(i * v for i, v in enumerate(p))), "probabilities": probs, "confidence": conf}
        return {"model": "Winnow-12B-npu-lpbq", "answers": answers,
                "usage": {"input_tokens": len(pre) + sum(len(s) for s in sufs), "output_tokens": 0},
                "npu": {"passes": len(packs), "ms": (time.perf_counter() - t0) * 1e3}}
