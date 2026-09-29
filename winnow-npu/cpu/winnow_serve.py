"""Start the patched Winnow server (llama-server + /v1/systemone) on the CPU only, 127.0.0.1.

Flags follow winnow-inference scripts/serve.py (pinned 6c2b3c0) minus the GPU parts: no layer offload, no CUDA/Metal
tensor override. Refuses to start below --min-free-gb of free RAM (the Q8_0 model alone is 12.7 GB).
    python winnow_serve.py [--port 8013] [--context 8192] [--threads 10] [--min-free-gb 14]
"""
import argparse, ctypes, os, subprocess, sys

W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8013); ap.add_argument("--context", type=int, default=8192)
ap.add_argument("--threads", type=int, default=10); ap.add_argument("--decision-parallel", type=int, default=4)
ap.add_argument("--min-free-gb", type=float, default=14.0)
ap.add_argument("--model", default=os.path.join(W, r"models\Winnow-12B\gguf\Winnow-12B-Q8_0.gguf"))
a, extra = ap.parse_known_args()   # unknown flags are passed through to winnow-server


class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
               ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]


m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
free = m.ullAvailPhys / 2**30
print(f"free RAM {free:.1f} GB (need {a.min_free_gb})", flush=True)
if free < a.min_free_gb:
    raise SystemExit("not enough free RAM; not loading Winnow")

exe = os.path.join(W, r"src\winnow-inference\.build\bin\winnow-server.exe")
env = dict(os.environ, WINNOW_CONTEXT=str(a.context), WINNOW_HEAD="selected", WINNOW_CACHE="q8_0", WINNOW_PIPELINE="optimized",
           WINNOW_MEMORY="auto", WINNOW_PARALLEL=str(a.decision_parallel), WINNOW_BATCH="2048", WINNOW_UBATCH="1024")
args = [exe, "--model", a.model, "--alias", "Winnow-12B", "--ctx-size", str(a.context), "--parallel", "1",
        "--n-gpu-layers", "0", "--fit", "off", "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--no-context-shift", "--lazy-mode", "off", "--batch-size", "2048", "--ubatch-size", "1024",
        "--threads", str(a.threads), "--host", "127.0.0.1", "--port", str(a.port), "--jinja", "--reasoning", "off",
        "--no-warmup", "--load-mode", "none", "--cache-ram", "0", "--cors-origins", "",
        # CPU weight repacking OFF: Winnow's selected answer head reads the tied embedding rows back with
        # ggml_backend_tensor_get, which the CPU repack buffer does not implement (null call -> access violation 0xc0000005)
        "--no-repack"]
proc = subprocess.Popen(args + extra, env=env)


class PPTS(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


# opt the server process itself out of EcoQoS throttling (a hidden background process is otherwise ~4x slower here)
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
st = PPTS(1, 0x1 | 0x4, 0)
ok = k32.SetProcessInformation(ctypes.c_void_p(int(proc._handle)), 4, ctypes.byref(st), ctypes.sizeof(st))
print(f"[winnow_serve] server pid {proc.pid}; power throttling disabled: {bool(ok)}", flush=True)
sys.exit(proc.wait())
