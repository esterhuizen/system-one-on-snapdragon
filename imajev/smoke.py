"""One guarded CPU smoke test of imajev-2b, text only (PyTorch CPU only; no GPU/NPU/llama.cpp)."""
import os, sys, time, json, ctypes
ROOT = os.path.expandvars(r"%USERPROFILE%\local-imajev\src\imajev")
os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="8")
class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
               ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
free_gb = m.ullAvailPhys / 2**30; NEED = float(os.environ.get("SMOKE_MIN_FREE_GB", "10.5")); print(f"free RAM {free_gb:.1f} GB (need {NEED})")
if free_gb < NEED: raise SystemExit("not enough free RAM; aborting before loading the model")
os.chdir(ROOT); sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts"), os.path.join(ROOT, "scripts", "playground")]
import torch; torch.set_num_threads(8)
from server import TorchBackend
from vision_decision.jev_api import to_request, to_response
t = time.perf_counter()
be = TorchBackend(bundle=os.path.join(ROOT, "artifacts", "model.json"), adapter=os.path.join(ROOT, "adapters", "imajev-2b"), device="cpu", rotations=1)
print(f"loaded in {time.perf_counter()-t:.1f}s")
body = {"state": {"document": "I was charged twice for my March invoice. Please refund the duplicate today."},
        "questions": {"billing": {"type": "noul", "instructions": "Is this ticket about billing?"},
                      "department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "charges, refunds", "technical": "bugs, outages", "sales": "pricing"}},
                      "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["not urgent", "soon", "today"]}}}
for i in range(2):
    req = to_request(body); t = time.perf_counter(); results, usage = be.score([], req); dt = (time.perf_counter() - t) * 1e3
    print(f"run {i}: {dt:.0f} ms")
print(json.dumps(to_response(req, results)["answers"], indent=1)[:1200])
