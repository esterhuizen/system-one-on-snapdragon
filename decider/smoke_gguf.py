"""Guarded smoke test of decider-2b GGUF on a CPU-only llama.cpp build (no GPU backend compiled; n_gpu_layers=0)."""
import os, sys, time, json, ctypes
os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
               ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
free_gb = m.ullAvailPhys / 2**30; print(f"free RAM {free_gb:.1f} GB")
if free_gb < 5: raise SystemExit("not enough free RAM (< 5 GB); aborting")
D = os.path.expandvars(r"%USERPROFILE%\local-decider\models\decider-2b-v11-gguf"); sys.path.insert(0, D)
from decide_gguf import GGUFDecider
q = os.environ.get("GGUF_QUANT", "Q8_0")
t = time.perf_counter(); d = GGUFDecider(os.path.join(D, f"decider-2b-v11-{q}.gguf"), n_ctx=2048, n_gpu_layers=0, n_threads=8); print(f"{q} loaded in {time.perf_counter()-t:.1f}s")
ctx = "I was charged twice for my March invoice. Please refund the duplicate today."
qs = [{"question": "Is this ticket about billing?", "options": ["yes", "no"]},
      {"question": "Which team should handle this?", "options": ["billing", "technical", "sales"]},
      {"question": "How urgent is it?", "options": ["not urgent", "soon", "today"]}]
for i in range(3):
    t = time.perf_counter(); r = d.decide(ctx, qs); print(f"run {i}: {(time.perf_counter()-t)*1e3:.0f} ms")
print(json.dumps(r, indent=1)[:900])
