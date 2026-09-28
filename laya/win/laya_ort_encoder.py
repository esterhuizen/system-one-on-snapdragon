"""Run Laya's ModernBERT encoder on an ONNX Runtime GPU EP (QNN GPU backend or DirectML) while the rest of
laya (tokenisation, typed decision head, calibration, Router, laya-serve Jev API) stays stock PyTorch-on-CPU.

Encoder models: static-shape exports  <dir>/encoder_b1_s{seq}_fp32g.onnx  for seq in BUCKETS
(see export_static_encoder.py + fix_allowzero.py + qnn_fix.py). Each laya row is padded to the next bucket
(attention_mask=0 on the pad) and run at batch 1; only the real positions are returned.
"""
import collections, os, types
import numpy as np
import torch
import onnxruntime as ort

PAD_ID = 50283  # ModernBERT [PAD]


def _verify(sess, want, path):
    """Refuse a session the accelerator EP is not actually running (no silent CPU-EP fallback)."""
    if want not in sess.get_providers():
        raise RuntimeError(f"{want} not active for {path}: providers={sess.get_providers()}")
    try:
        info = sess.get_provider_graph_assignment_info()
    except AttributeError:
        return sess
    n = collections.Counter()
    for sg in info:
        n[sg.ep_name] += len(sg.get_nodes())
    print(f"[laya_ort_encoder] {os.path.basename(path)} placement={dict(n)}", flush=True)
    if not n.get(want):
        raise RuntimeError(f"{want} was assigned 0 nodes of {path}")
    return sess


def _session(path, backend):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    if backend == "qnngpu":
        import onnxruntime_qnn as q
        if not any(d.ep_name == "QNNExecutionProvider" for d in ort.get_ep_devices()):
            ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
        devs = [d for d in ort.get_ep_devices()
                if d.ep_name == "QNNExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.GPU]
        if not devs:
            raise RuntimeError("QNN GPU device not found")
        so.add_provider_for_devices(devs, {})
        return _verify(ort.InferenceSession(path, so), "QNNExecutionProvider", path)
    if backend == "dml":
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return _verify(ort.InferenceSession(path, so, providers=["DmlExecutionProvider"]), "DmlExecutionProvider", path)
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class OrtEncoder(torch.nn.Module):
    def __init__(self, model_dir, backend="qnngpu", buckets=(128, 256, 512), tag="fp32g", hidden=1024):
        super().__init__()
        self.backend = backend
        self.buckets = sorted(b for b in buckets if os.path.exists(os.path.join(model_dir, f"encoder_b1_s{b}_{tag}.onnx")))
        if not self.buckets:
            raise FileNotFoundError(f"no encoder_b1_s*_{tag}.onnx in {model_dir}")
        self.sessions = {b: _session(os.path.join(model_dir, f"encoder_b1_s{b}_{tag}.onnx"), backend) for b in self.buckets}
        self.config = types.SimpleNamespace(hidden_size=hidden, reference_compile=False)
        self.hidden = hidden

    def forward(self, input_ids, attention_mask, **_):
        B, L = input_ids.shape
        bucket = next((b for b in self.buckets if b >= L), None)
        if bucket is None:
            raise ValueError(f"sequence {L} longer than largest bucket {self.buckets[-1]}")
        ids_np = input_ids.cpu().numpy().astype(np.int64)
        am_np = attention_mask.cpu().numpy().astype(np.int64)
        out = np.zeros((B, L, self.hidden), np.float32)
        sess = self.sessions[bucket]
        for i in range(B):
            ids = np.full((1, bucket), PAD_ID, np.int64); ids[0, :L] = ids_np[i]
            am = np.zeros((1, bucket), np.int64); am[0, :L] = am_np[i]
            out[i] = sess.run(None, {"input_ids": ids, "attention_mask": am})[0][0, :L]
        return types.SimpleNamespace(last_hidden_state=torch.from_numpy(out))


def to_gpu(agent, model_dir, backend="qnngpu", **kw):
    """Swap agent.model.encoder for the ORT encoder in place (frees the torch encoder weights)."""
    agent.model.encoder = OrtEncoder(model_dir, backend=backend, **kw)
    return agent
