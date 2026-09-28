"""Prove the static Laya encoder buckets execute on the Adreno (QNN GPU or DirectML); parity + latency vs ORT CPU.
python -X utf8 gpu_proof.py --backend qnngpu [--tag fp32g] [--seqs 128,512] [-n 50]"""
import argparse, collections, os, time
import numpy as np, onnxruntime as ort
ap = argparse.ArgumentParser(); ap.add_argument("--backend", required=True, choices=["qnngpu", "dml"])
ap.add_argument("--enc-dir", default=os.path.expandvars(r"%USERPROFILE%\local-laya\models\gpu-enc")); ap.add_argument("--tag", default="fp32g")
ap.add_argument("--seqs", default="128,512"); ap.add_argument("-n", type=int, default=50); a = ap.parse_args()
ort.set_default_logger_severity(1)   # each 'backendValidateOpConfig() failed for node X' = one node QNN rejected (runs on CPU)
_reg = [False]
def accel(path):
    so = ort.SessionOptions(); so.log_severity_level = 1
    so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    if a.backend == "qnngpu":
        import onnxruntime_qnn as q
        if not _reg[0]: ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path()); _reg[0] = True
        devs = [x for x in ort.get_ep_devices() if x.ep_name == "QNNExecutionProvider" and x.device.type == ort.OrtHardwareDeviceType.GPU]
        if not devs: raise SystemExit("QNN EP exposes no GPU device")
        print("[device]", devs[0].device.metadata.get("Description"), flush=True)
        so.add_provider_for_devices(devs, {}); return ort.InferenceSession(path, so)
    so.enable_mem_pattern = False; so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(path, so, providers=["DmlExecutionProvider"])
def bench(s, feed, n):
    for _ in range(5): s.run(None, feed)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); s.run(None, feed); ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts); return f"p50 {np.percentile(ts,50):.1f} ms p95 {np.percentile(ts,95):.1f} ms min {ts.min():.1f} ms n={n}"
for seq in map(int, a.seqs.split(",")):
    path = os.path.join(a.enc_dir, f"encoder_b1_s{seq}_{a.tag}.onnx"); L = seq * 3 // 4
    rng = np.random.default_rng(seq); ids = np.full((1, seq), 50283, np.int64); ids[0, :L] = rng.integers(1000, 50000, L)
    ids[0, 0], ids[0, L - 1] = 50281, 50282; am = np.zeros((1, seq), np.int64); am[0, :L] = 1
    feed = {"input_ids": ids, "attention_mask": am}
    t0 = time.perf_counter(); s = accel(path); print(f"[s{seq}] session {time.perf_counter()-t0:.1f}s providers={s.get_providers()}", flush=True)
    try:
        for sg in s.get_provider_graph_assignment_info():
            ops = collections.Counter(n.op_type for n in sg.get_nodes())
            print(f"[s{seq}] [placement] {sg.ep_name}: {sum(ops.values())} nodes {dict(ops) if 'CPU' in sg.ep_name else ''}", flush=True)
    except AttributeError:
        print("[placement] API unavailable; rely on INFO log lines", flush=True)
    cpu = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ref = cpu.run(None, feed)[0][0, :L].astype(np.float32); out = s.run(None, feed)[0][0, :L].astype(np.float32)
    cos = (ref * out).sum(-1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(out, axis=-1) + 1e-9)
    print(f"[s{seq}] [parity] finite={np.isfinite(out).all()} min_token_cos={cos.min():.6f} max|d|={np.abs(ref-out).max():.2e}", flush=True)
    print(f"[s{seq}] [latency] {a.backend}: {bench(s, feed, a.n)}", flush=True); print(f"[s{seq}] [latency] cpu-ep: {bench(cpu, feed, 10)}", flush=True)
    del s, cpu
