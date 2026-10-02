"""decider-12b (Mapika, Gemma-4-12B-it + LoRA) on the Snapdragon Hexagon NPU: Jev-compatible decisions.

Prompt, rows and answers are decider-ai's own (decider.serve.prepare with the chat layout, decider.systemone.assemble,
decider_config.json temperatures by type). The 48 decoder layers run as 12 LPBQ-int4 QNN chunks (built with the Winnow
pipeline: winnow-npu/scripts); each request packs its question rows after one shared prefix (template head + context),
with a block attention mask and per-row positions, exactly like the per-row forwards decider.serve runs. The answer slot is
the last token of each row: logits = softcap30(RMSNorm(h) . token_embd[letter]) / T_type, softmax over the options.

    rt = DeciderNPU(chain="chain_decider12b"); rt.decide({"state": ..., "questions": {...}})
Run from ONE thread (QNN sessions called from many threads silently corrupt results: onnxruntime-qnn#892).
"""
import json, math, os, sys, threading, time
import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_qnn as oq
from transformers import AutoTokenizer
from decider import serve as SV, systemone as S1, temperature as TT
from decider.prompt import chat_template, letter_ids

WW = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MD = os.environ.get("DECIDER_MODELS") or os.path.join(WW, "models")   # decider-12b (HF files) + decider-12b-gguf
sys.path.insert(0, os.path.join(WW, "src", "winnow-inference", ".runtime", "llama.cpp", "gguf-py"))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

D, CAP, MASK = 3840, 30.0, -60.0


class DeciderNPU:
    def __init__(self, chain="chain_decider12b", seq=576, variant="lpbq32", model_dir=os.path.join(MD, "decider-12b"),
                 gguf_path=os.path.join(MD, "decider-12b-gguf", "decider-12b-v2-Q8_0.gguf"), perf="burst"):
        self.S = seq; mp = "model_packed" if seq == 512 else f"model_packed{seq}"
        SV.apply_config(json.load(open(os.path.join(model_dir, "decider_config.json"))))
        self.tok = AutoTokenizer.from_pretrained(model_dir); self.chat = chat_template(self.tok)
        g = gguf.GGUFReader(gguf_path); T = {t.name: t for t in g.tensors}; kv = lambda k: g.fields[k].contents()
        E = T["token_embd.weight"]; self._raw, self._et = np.asarray(E.data).reshape(int(E.shape[1]), -1), E.tensor_type
        self.onorm = np.asarray(T["output_norm.weight"].data, np.float32); self.eps = kv("gemma4.attention.layer_norm_rms_epsilon")
        self.letters = self.emb(letter_ids(self.tok)[:64])
        rf = np.asarray(T["rope_freqs.weight"].data, np.float64)
        self.inv = {"swa": kv("gemma4.rope.freq_base_swa") ** (-np.arange(128, dtype=np.float64) * 2 / 256),
                    "glb": kv("gemma4.rope.freq_base") ** (-np.arange(256, dtype=np.float64) * 2 / 512) / rf}
        if not any(x.ep_name == oq.EP_NAME for x in ort.get_ep_devices()):
            ort.register_execution_provider_library(oq.EP_NAME, oq.get_library_path())
        npu = [x for x in ort.get_ep_devices() if x.ep_name == oq.EP_NAME and x.device.type == ort.OrtHardwareDeviceType.NPU]
        self.sess = []
        for k in range(12):
            d = os.path.join(WW, "onnx", chain, f"c{k:02d}_{variant}"); ctx = os.path.join(d, mp + "_ctx.onnx")
            if not os.path.exists(ctx):
                raise FileNotFoundError(f"{ctx} missing: compile first")
            so = ort.SessionOptions(); so.log_severity_level = 3; so.add_provider_for_devices(npu, {"htp_performance_mode": perf})
            s = ort.InferenceSession(ctx, so)
            m = onnx.load(os.path.join(d, mp + ".onnx"), load_external_data=False); I = {i.name: i for i in m.graph.initializer}
            qp = {i.name: (i.name[:-10], float(onnx.numpy_helper.to_array(I[i.name[:-10] + "_scale"])), int(onnx.numpy_helper.to_array(I[i.name[:-10] + "_zero_point"])))
                  for i in s.get_inputs() if i.name != "x"}
            self.sess.append((s, qp))
        self.lock = threading.Lock()

    def emb(self, ids):
        return dequantize(self._raw[np.asarray(ids)], self._et).reshape(len(ids), D).astype(np.float32)

    def _tables(self, prefix_len, q_lens):
        S = self.S; pos = np.zeros(S, np.float64); grp = np.full(S, -1)
        pos[:prefix_len] = np.arange(prefix_len); grp[:prefix_len] = 0; o = prefix_len
        for qi, L in enumerate(q_lens, 1):
            pos[o: o + L] = prefix_len + np.arange(L); grp[o: o + L] = qi; o += L
        pos[o:] = np.arange(S - o)
        i, j = np.arange(S)[:, None], np.arange(S)[None, :]
        ok = ((j <= i) & ((grp[None, :] == 0) | (grp[None, :] == grp[:, None]))) | (i == j)
        mask = np.where(ok, 0.0, MASK).astype(np.float32)[None]; t = {"mask_swa": mask, "mask_glb": mask}
        for kind in ("swa", "glb"):
            ang = pos[:, None] * self.inv[kind][None, :]
            t[f"cos_{kind}"] = np.concatenate([np.cos(ang)] * 2, -1).astype(np.float32)[:, None, :]
            t[f"sin_{kind}"] = np.concatenate([np.sin(ang)] * 2, -1).astype(np.float32)[:, None, :]
        return t

    def _pass(self, prefix, suffixes):
        ids = prefix + [t for s in suffixes for t in s]
        x = np.zeros((1, self.S, D), np.float32); x[0, : len(ids)] = self.emb(ids) * math.sqrt(D)
        tab = self._tables(len(prefix), [len(s) for s in suffixes])
        for s, qp in self.sess:
            feed = {"x": x}
            for name, (base, sc, zp) in qp.items():
                feed[name] = np.clip(np.round(tab[base] / sc) + zp, 0, 65535).astype(np.uint16)
            x = s.run(["y"], feed)[0]
        rows, end = [], len(prefix)
        for suf in suffixes:
            end += len(suf); h = x[0, end - 1]; rows.append(h / np.sqrt((h * h).mean() + self.eps) * self.onorm)
        return rows

    def decide(self, body):
        rqs, index, items, ctx_len = SV.prepare(self.tok, body["state"], body["questions"], bool(body.get("independent", True)),
                                                SV.ISOLATED, 32768, chat=self.chat)
        prefix = items[0]["ids"][:ctx_len]; sufs = []
        for it in items:
            assert it["ids"][:ctx_len] == prefix and it["slots"] == [len(it["ids"]) - 1], "unexpected row layout"
            if len(it["ids"]) > self.S:
                raise ValueError(f"state + question is {len(it['ids'])} tokens > {self.S}")
            sufs.append(it["ids"][ctx_len:])
        packs, cur, used = [], [], ctx_len
        for i, s in enumerate(sufs):
            if cur and used + len(s) > self.S: packs.append(cur); cur, used = [], ctx_len
            cur.append(i); used += len(s)
        packs.append(cur)
        probs, raw, t0 = [None] * len(items), [None] * len(items), time.perf_counter()
        with self.lock:
            for pk in packs:
                for i, h in zip(pk, self._pass(prefix, [sufs[i] for i in pk])):
                    n = items[i]["nopts"][0]; T = TT.for_types(SV.TEMP, SV.TEMP_BY_TYPE, items[i]["types"])
                    T = T[0] if isinstance(T, list) else T
                    raw[i] = CAP * np.tanh((self.letters[:n] @ h) / CAP)
                    lg = raw[i] / T; p = np.exp(lg - lg.max()); probs[i] = (p / p.sum()).tolist()
        answers = S1.assemble(rqs, index, probs)
        if os.environ.get("DECIDER_NPU_RAW"):   # analysis only: soft-capped letter logits before the temperature
            for k, kind, s, n in index:
                if kind != "iso": answers[k]["x_raw_logits"] = [round(float(v), 4) for v in raw[s]]
        return {"model": SV.MODEL_NAME + "-npu-lpbq", "answers": answers,
                "usage": {"input_tokens": ctx_len + sum(len(s) for s in sufs), "output_tokens": 0},
                "npu": {"passes": len(packs), "ms": (time.perf_counter() - t0) * 1e3}}
