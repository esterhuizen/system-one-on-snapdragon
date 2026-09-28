"""Find which part of the exported graph fails QNN-GPU finalize: cut the (topologically ordered) node list into chunks,
build each chunk as a standalone model (boundary tensors become inputs/outputs, weights referenced in place), and only create
a QNN GPU session (no inference).  python qnn_bisect.py MODEL.onnx --chunks 40 [--range a b]"""
import argparse, os, sys, time, json, collections
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort, onnxruntime_qnn as q
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--chunks", type=int, default=40)
ap.add_argument("--range", type=int, nargs=2); ap.add_argument("--bisect", action="store_true"); a = ap.parse_args()
d = os.path.dirname(a.model); m = onnx.load(a.model, load_external_data=False)
m = onnx.shape_inference.infer_shapes(m, check_type=False, strict_mode=False, data_prop=False)
g = m.graph; N = len(g.node)
vi = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}
inits = {i.name: i for i in g.initializer}
ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
devs = [x for x in ort.get_ep_devices() if x.ep_name == "QNNExecutionProvider" and x.device.type == ort.OrtHardwareDeviceType.GPU]
cons_after = collections.defaultdict(set)
for i, n in enumerate(g.node):
    for t in n.input: cons_after[t].add(i)
graph_outs = {o.name for o in g.output}
def build(a0, b0):
    nodes = g.node[a0:b0]; produced = {o for n in nodes for o in n.output}
    need = []; seen = set()
    for n in nodes:
        for t in n.input:
            if t and t not in produced and t not in seen:
                seen.add(t); need.append(t)
    ins, used_inits = [], []
    for t in need:
        if t in inits: used_inits.append(inits[t])
        elif t in vi: ins.append(vi[t])
        else: return None, f"no type info for boundary tensor {t}"
    outs = []
    for n in nodes:
        for o in n.output:
            if o and (o in graph_outs or any(c >= b0 for c in cons_after.get(o, ()))):
                if o not in vi: return None, f"no type info for output {o}"
                outs.append(vi[o])
    if not outs: outs = [vi[nodes[-1].output[0]]] if nodes[-1].output[0] in vi else []
    sub = helper.make_graph(nodes, f"chunk_{a0}_{b0}", ins, outs, initializer=used_inits)
    mm = helper.make_model(sub, opset_imports=m.opset_import, ir_version=m.ir_version)
    p = os.path.join(d, f"_chunk_{a0}_{b0}.onnx"); onnx.save(mm, p); return p, None
def test(a0, b0):
    p, err = build(a0, b0)
    if err: return "skip", err
    so = ort.SessionOptions(); so.log_severity_level = 3; so.add_provider_for_devices(devs, {})
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
    t = time.perf_counter()
    try:
        s = ort.InferenceSession(p, so); ok, msg = "ok", f"{time.perf_counter()-t:.1f}s"
    except Exception as e:
        ok, msg = "FAIL", str(e).split("\n")[0][-160:]
    finally:
        try: os.remove(p)
        except OSError: pass
    return ok, msg
def ops(a0, b0): return dict(collections.Counter(n.op_type for n in g.node[a0:b0]).most_common(8))
if a.range:
    print(a.range, *test(*a.range), ops(*a.range), flush=True); sys.exit()
step = (N + a.chunks - 1) // a.chunks; fails = []
for a0 in range(0, N, step):
    b0 = min(N, a0 + step); r, msg = test(a0, b0); print(f"[{a0:5d},{b0:5d}) {r:4s} {msg}", flush=True)
    if r == "FAIL": fails.append((a0, b0))
print("failing chunks:", fails, flush=True)
if a.bisect:
    for a0, b0 in fails[:3]:
        lo, hi = a0, b0
        while hi - lo > 1:
            mid = (lo + hi) // 2; r1, _ = test(lo, mid); r2, _ = test(mid, hi)
            if r1 == "FAIL": hi = mid
            elif r2 == "FAIL": lo = mid
            else: print(f"  failure in [{lo},{hi}) needs nodes from both halves", flush=True); break
        n = g.node[lo]; print(f"  minimal failing range [{lo},{hi}): {ops(lo, hi)}  first node {n.op_type} {n.name} inputs {list(n.input)[:3]}", flush=True)
