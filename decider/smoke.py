"""One guarded CPU smoke test of decider-2b (PyTorch CPU only; no GPU/NPU/llama.cpp)."""
import os, time, json, ctypes
os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DECIDER_DEVICE="cpu", OMP_NUM_THREADS="8")
class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
               ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
free_gb = m.ullAvailPhys / 2**30; print(f"free RAM {free_gb:.1f} GB")
NEED = float(os.environ.get("SMOKE_MIN_FREE_GB", "8"))
if free_gb < NEED: raise SystemExit("not enough free RAM (< 8 GB); aborting before loading the model")
import torch; torch.set_num_threads(8)
from decider.infer import Decider
t = time.perf_counter(); d = Decider(os.path.expandvars(r"%USERPROFILE%\local-decider\models\decider-2b-v11"), dtype=getattr(torch, os.environ.get("SMOKE_DTYPE", "bfloat16"))); print(f"loaded in {time.perf_counter()-t:.1f}s on {getattr(d,'device','?')}")
state = {"document": "I was charged twice for my March invoice. Please refund the duplicate today."}
qs = {"billing": {"type": "noul", "instructions": "Is this ticket about billing?"},
      "department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "charges, refunds", "technical": "bugs, outages", "sales": "pricing"}},
      "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["not urgent", "soon", "today"]}}
for i in range(3):
    t = time.perf_counter(); r = d.system_one(state, qs); dt = (time.perf_counter() - t) * 1e3
    print(f"run {i}: {dt:.0f} ms")
print(json.dumps(r, indent=1)[:900])
