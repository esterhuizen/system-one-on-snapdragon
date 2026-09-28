"""decider-2b on ONNX Runtime (static S=256, K=8 slots), reusing decider's prompt building and temperatures.
python onnx_decider.py --ep cpu|qnngpu"""
import os, sys, time, json, ctypes, argparse, collections
os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
ap = argparse.ArgumentParser(); ap.add_argument("--ep", default="cpu"); ap.add_argument("--min-free-gb", type=float, default=6.0)
ap.add_argument("--ops", action="store_true"); a = ap.parse_args()
class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
               ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
print(f"free RAM {m.ullAvailPhys/2**30:.1f} GB", flush=True)
if m.ullAvailPhys / 2**30 < a.min_free_gb: raise SystemExit("not enough free RAM; aborting")
import numpy as np, onnxruntime as ort
from transformers import AutoTokenizer
from decider.infer import Example, Q, _NoShuffle
from decider.prompt import MAX_OPTIONS, build
from decider import temperature as TT
M = os.path.expandvars(r"%USERPROFILE%\local-decider\models\decider-2b-v11"); G = os.path.expandvars(r"%USERPROFILE%\local-decider\models\decider-2b-onnx\decider2b_s256_k8_") + os.environ.get("DECIDER_ONNX_TAG", "float16") + ".onnx"
S, K = 256, 8
if a.ops:
    import onnx; g = onnx.load(G, load_external_data=False)
    print("ops:", dict(collections.Counter(n.op_type for n in g.graph.node).most_common())); raise SystemExit
tok = AutoTokenizer.from_pretrained(M); cfg = json.load(open(os.path.join(M, "decider_config.json"))); (T0, Tby), _ = TT.from_config(cfg)
so = ort.SessionOptions(); so.log_severity_level = int(os.environ.get("ORT_LOG", "3")); so.intra_op_num_threads = 8
so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
t = time.perf_counter()
if a.ep == "qnngpu":
    import onnxruntime_qnn as q; ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.GPU]
    so.add_provider_for_devices(devs, {}); sess = ort.InferenceSession(G, so)
else:
    sess = ort.InferenceSession(G, so, providers=["CPUExecutionProvider"])
print(f"session ({a.ep}) {time.perf_counter()-t:.1f}s providers={sess.get_providers()}", flush=True)
try:
    for sg in sess.get_provider_graph_assignment_info():
        c = collections.Counter(n.op_type for n in sg.get_nodes()); print(f"[placement] {sg.ep_name}: {sum(c.values())} nodes", dict(c) if 'CPU' in sg.ep_name and a.ep != 'cpu' else '')
except Exception as e: print("[placement] n/a", e)
def decide(context, questions):
    item = build(Example(context, [Q(x["question"], list(x["options"]), 0) for x in questions]), tok, _NoShuffle(), max_options=MAX_OPTIONS, max_ctx_tokens=1536)
    ids = np.full((1, S), tok.pad_token_id or 0, np.int64); n = len(item["ids"]); assert n <= S and len(item["slots"]) <= K, (n, len(item["slots"]))
    ids[0, :n] = item["ids"]; sl = np.zeros(K, np.int64); sl[:len(item["slots"])] = item["slots"]
    lg = sess.run(["logits"], {"input_ids": ids, "slot_idx": sl})[0]
    out = []
    for i, (qq, nopt, tt) in enumerate(zip(questions, item["nopts"], TT.for_types(T0, Tby, ["choice"] * len(questions)))):
        z = lg[i, :nopt].astype(np.float64) / tt; p = np.exp(z - z.max()); p /= p.sum(); j = int(p.argmax())
        out.append({"choice": qq["options"][j], "p": round(float(p[j]), 4), "probs": dict(zip(qq["options"], np.round(p, 4).tolist()))})
    return out, n
ctx = "I was charged twice for my March invoice. Please refund the duplicate today."
qs = [{"question": "Is this ticket about billing?", "options": ["yes", "no"]},
      {"question": "Which team should handle this?", "options": ["billing", "technical", "sales"]},
      {"question": "How urgent is it?", "options": ["not urgent", "soon", "today"]}]
for i in range(3):
    t = time.perf_counter(); r, n = decide(ctx, qs); print(f"run {i}: {(time.perf_counter()-t)*1e3:.0f} ms  ({n} tokens)", flush=True)
print(json.dumps(r))
print("GGUF Q8_0 reference: yes 0.9736 | billing 0.9878 | today 0.8649")
