"""Export decider-2b (backbone + option-letter readout) to a static ONNX graph: input_ids[1,S], slot_idx[K] -> logits[K, MAX_OPTIONS].
Right-padded causal prompts: tokens after a slot cannot change it, so no attention mask is needed."""
import os, sys, time, ctypes, argparse
os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
ap = argparse.ArgumentParser(); ap.add_argument("--seq", type=int, default=256); ap.add_argument("--slots", type=int, default=8)
ap.add_argument("--dtype", default="float16"); ap.add_argument("--min-free-gb", type=float, default=8.0); a = ap.parse_args()
class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
               ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
free_gb = m.ullAvailPhys / 2**30; print(f"free RAM {free_gb:.1f} GB", flush=True)
if free_gb < a.min_free_gb: raise SystemExit("not enough free RAM; aborting")
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from decider.prompt import letter_ids, MAX_OPTIONS
torch.set_num_threads(8)
M = os.path.expandvars(r"%USERPROFILE%\local-decider\models\decider-2b-v11")
OUT = os.path.expandvars(rf"%USERPROFILE%\local-decider\models\decider-2b-onnx\decider2b_s{a.seq}_k{a.slots}_{a.dtype}.onnx")
tok = AutoTokenizer.from_pretrained(M)
t = time.perf_counter(); lm = AutoModelForCausalLM.from_pretrained(M, dtype=getattr(torch, a.dtype)).eval(); print(f"loaded {time.perf_counter()-t:.0f}s", flush=True)
class Readout(nn.Module):
    def __init__(s, lm, letters):
        super().__init__(); s.backbone = lm.model; s.register_buffer("Wl", lm.lm_head.weight[letters].detach().clone())
    def forward(s, input_ids, slot_idx):
        h = s.backbone(input_ids=input_ids, use_cache=False).last_hidden_state[0]
        return F.linear(h[slot_idx], s.Wl).float()
w = Readout(lm, torch.tensor(letter_ids(tok))).eval(); assert w.Wl.shape[0] == MAX_OPTIONS
ids = torch.full((1, a.seq), tok.pad_token_id or 0, dtype=torch.long); ids[0, :40] = torch.randint(100, 20000, (40,))
slots = torch.tensor([39] + [0] * (a.slots - 1), dtype=torch.long)
t = time.perf_counter()
with torch.no_grad():
    torch.onnx.export(w, (ids, slots), OUT, input_names=["input_ids", "slot_idx"], output_names=["logits"], dynamo=True, external_data=True, optimize=True)
print(f"exported {OUT} in {time.perf_counter()-t:.0f}s", flush=True)
