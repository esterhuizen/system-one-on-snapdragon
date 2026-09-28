"""Split the exported decider graph into consecutive segments (<= --max-nodes each, the QNN GPU finalize limit is ~3000 nodes),
save them next to the model (weights referenced in place), and run them as a chain of QNN GPU sessions.
python qnn_chain.py MODEL.onnx --max-nodes 2500 [--ep qnngpu|cpu]"""
import argparse, os, sys, time, json, collections, ctypes
import numpy as np, onnx
from onnx import helper
import onnxruntime as ort
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--max-nodes", type=int, default=2500); ap.add_argument("--ep", default="qnngpu"); a = ap.parse_args()
d = os.path.dirname(a.model); tag = os.path.splitext(os.path.basename(a.model))[0]
m = onnx.shape_inference.infer_shapes(onnx.load(a.model, load_external_data=False), check_type=False, strict_mode=False)
g = m.graph; N = len(g.node); vi = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}; inits = {i.name: i for i in g.initializer}
last_use = {}
for i, n in enumerate(g.node):
    for t in n.input: last_use[t] = i
graph_outs = [o.name for o in g.output]
segs, paths = [(s, min(N, s + a.max_nodes)) for s in range(0, N, a.max_nodes)], []
for k, (s, e) in enumerate(segs):
    nodes = g.node[s:e]; produced = {o for n in nodes for o in n.output}; need, seen = [], set()
    for n in nodes:
        for t in n.input:
            if t and t not in produced and t not in seen: seen.add(t); need.append(t)
    ins = [vi[t] for t in need if t not in inits]; used = [inits[t] for t in need if t in inits]
    outs = [vi[o] for n in nodes for o in n.output if o and (o in graph_outs or last_use.get(o, -1) >= e)]
    p = os.path.join(d, f"{tag}_seg{k}.onnx")
    onnx.save(helper.make_model(helper.make_graph(nodes, f"seg{k}", ins, outs, initializer=used), opset_imports=m.opset_import, ir_version=m.ir_version), p)
    paths.append(p)
print(f"{len(paths)} segments of <= {a.max_nodes} nodes", flush=True)
so_base = lambda: ort.SessionOptions()
if a.ep == "qnngpu":
    import onnxruntime_qnn as q; ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
    devs = [x for x in ort.get_ep_devices() if x.ep_name == "QNNExecutionProvider" and x.device.type == ort.OrtHardwareDeviceType.GPU]
sess = []; t = time.perf_counter()
for p in paths:
    so = ort.SessionOptions(); so.log_severity_level = 3; so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
    if a.ep == "qnngpu": so.add_provider_for_devices(devs, {}); s = ort.InferenceSession(p, so)
    else: s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
    pl = collections.Counter()
    try:
        for sg in s.get_provider_graph_assignment_info(): pl[sg.ep_name] += len(sg.get_nodes())
    except Exception: pass
    sess.append(s); print(f"  {os.path.basename(p)}: {dict(pl)}", flush=True)
print(f"sessions ready in {time.perf_counter()-t:.0f}s", flush=True)
def run_chain(feed):
    env = dict(feed)
    for s in sess:
        outs = s.run(None, {i.name: env[i.name] for i in s.get_inputs()})
        env.update({o.name: v for o, v in zip(s.get_outputs(), outs)})
    return env["logits"]
os.environ.update(HF_HUB_OFFLINE="1")
from transformers import AutoTokenizer
from decider.infer import Example, Q, _NoShuffle
from decider.prompt import MAX_OPTIONS, build
from decider import temperature as TT
M = os.path.expandvars(r"%USERPROFILE%\local-decider\models\decider-2b-v11")
tok = AutoTokenizer.from_pretrained(M); (T0, Tby), _ = TT.from_config(json.load(open(os.path.join(M, "decider_config.json"))))
ctx = "I was charged twice for my March invoice. Please refund the duplicate today."
qs = [{"question": "Is this ticket about billing?", "options": ["yes", "no"]},
      {"question": "Which team should handle this?", "options": ["billing", "technical", "sales"]},
      {"question": "How urgent is it?", "options": ["not urgent", "soon", "today"]}]
item = build(Example(ctx, [Q(x["question"], x["options"], 0) for x in qs]), tok, _NoShuffle(), max_options=MAX_OPTIONS, max_ctx_tokens=1536)
ids = np.full((1, 256), tok.pad_token_id or 0, np.int64); ids[0, :len(item["ids"])] = item["ids"]; sl = np.zeros(8, np.int64); sl[:len(item["slots"])] = item["slots"]
for i in range(3):
    t = time.perf_counter(); lg = run_chain({"input_ids": ids, "slot_idx": sl}); print(f"run {i}: {(time.perf_counter()-t)*1e3:.0f} ms", flush=True)
res = []
for i, (x, n, tt) in enumerate(zip(qs, item["nopts"], TT.for_types(T0, Tby, ["choice"] * 3))):
    z = lg[i, :n].astype(np.float64) / tt; p = np.exp(z - z.max()); p /= p.sum(); j = int(p.argmax()); res.append((x["options"][j], round(float(p[j]), 4)))
print("answers:", res, "| GGUF Q8_0 reference: yes 0.9736, billing 0.9878, today 0.8649", flush=True)
