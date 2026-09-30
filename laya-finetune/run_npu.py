"""Run an exported classifier (export_npu.py *_q.onnx) on the Hexagon NPU (QNN HTP, fp16) and write predictions with
per-item timing; optionally score against labels and compare with ONNX Runtime on the CPU.
Needs onnxruntime 1.30 + onnxruntime-qnn 2.6.0 (plugin EP) and transformers. Compiles once to an EPContext cache.
Call the NPU from ONE thread (see docs/WINNOW-NPU.md "The traps").

python run_npu.py --model OUT/onnx/classifier_q.onnx --ckpt-dir OUT --items items.jsonl --out preds.jsonl
                  [--ids ids.txt] [--gold labels.jsonl] [--compare-cpu]
"""
import argparse, collections, json, os, time
import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as oq
from transformers import AutoTokenizer

try:
    import nothrottle  # noqa: F401
except ImportError:
    pass
ap = argparse.ArgumentParser()
for k in ("model", "ckpt-dir", "items", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--ids"); ap.add_argument("--gold"); ap.add_argument("--compare-cpu", action="store_true"); a = ap.parse_args()
cfg = json.load(open(os.path.join(a.ckpt_dir, "classifier_config.json"))); TASKS = cfg["tasks"]; S = cfg["max_len"]
tok = AutoTokenizer.from_pretrained(os.path.join(cfg["snapshot"], "tokenizer"))


class _Blank(dict):
    def __missing__(self, k): return ""


text = lambda it: cfg["text_template"].format_map(_Blank({k: ("" if v is None else v) for k, v in it.items()}))
items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
if a.ids:
    keep = set(open(a.ids).read().split()); items = [it for it in items if it["id"] in keep]
gold = {r["id"]: r for r in map(json.loads, open(a.gold, encoding="utf-8"))} if a.gold else {}
ort.register_execution_provider_library(oq.EP_NAME, oq.get_library_path())
npu_dev = [d for d in ort.get_ep_devices() if d.ep_name == oq.EP_NAME and d.device.type == ort.OrtHardwareDeviceType.NPU]
assert npu_dev, "no QNN NPU device"
ctx = a.model.replace(".onnx", "_npu_ctx.onnx")
so = ort.SessionOptions(); so.log_severity_level = 3; so.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
if not os.path.exists(ctx):
    so.add_session_config_entry("ep.context_enable", "1"); so.add_session_config_entry("ep.context_file_path", ctx)
    so.add_session_config_entry("ep.context_embed_mode", "0")
so.add_provider_for_devices(npu_dev, {"htp_performance_mode": "burst", "enable_htp_fp16_precision": "1"})
t = time.perf_counter(); npu = ort.InferenceSession(ctx if os.path.exists(ctx) else a.model, so)
place = collections.Counter()
for sg in npu.get_provider_graph_assignment_info(): place[sg.ep_name] += sum(1 for _ in sg.get_nodes())
print(f"NPU session in {time.perf_counter() - t:.0f}s, placement {dict(place)}", flush=True)
cpu = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"]) if a.compare_cpu else None
ok = collections.Counter(); n = collections.Counter(); same = collections.Counter(); ts = []
with open(a.out, "w", encoding="utf-8") as fo:
    for it in items:
        t1 = time.perf_counter()
        b = tok([text(it)], padding="max_length", truncation=True, max_length=S, return_tensors="np")
        feed = {"input_ids": b["input_ids"].astype(np.int64), "attention_mask": b["attention_mask"].astype(np.int64)}
        t2 = time.perf_counter(); outs = npu.run(None, feed); t3 = time.perf_counter(); ts.append((t3 - t1) * 1e3)
        rec = {"id": it["id"], "ms_total": (t3 - t1) * 1e3, "ms_npu": (t3 - t2) * 1e3}
        cpu_outs = cpu.run(None, feed) if cpu else None
        for k, (tname, keys) in enumerate(TASKS.items()):
            p = outs[k][0]; j = int(p.argmax())
            rec[tname] = keys[j]; rec[f"p_{tname}"] = float(p[j]); rec[f"{tname}_probs"] = dict(zip(keys, p.tolist()))
            if cpu_outs is not None: same[tname] += int(cpu_outs[k][0].argmax() == j)
            if gold.get(it["id"], {}).get(tname): n[tname] += 1; ok[tname] += gold[it["id"]][tname] == keys[j]
        fo.write(json.dumps(rec) + "\n")
print(f"{len(items)} items; median {np.median(ts):.1f} ms per item (tokenise + NPU), p95 {np.percentile(ts, 95):.1f} ms -> {a.out}")
for tname in TASKS:
    if n[tname]: print(f"  {tname}: accuracy vs labels {ok[tname] / n[tname]:.1%} (n={n[tname]})")
    if cpu: print(f"  {tname}: same answer as ORT CPU {same[tname]}/{len(items)}")
