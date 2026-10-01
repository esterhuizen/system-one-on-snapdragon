"""NPU chain vs PyTorch reference on the six Decider-format rows: soft-capped answer-letter logits."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_decider import DeciderNPU, CAP
M = os.environ.get("DECIDER_MODELS") or "models"
R = json.load(open(os.path.join(M, "decider12b_refrows.json"))); T = json.load(open(os.path.join(M, "decider12b_torch_ref.json")))
t = time.perf_counter(); rt = DeciderNPU(); print(f"loaded in {time.perf_counter() - t:.0f}s", flush=True)
same = 0; dl = []
for r, ref in zip(R["rows"], T):
    t = time.perf_counter(); h = rt._pass([], [r["ids"]])[0]; ms = (time.perf_counter() - t) * 1e3
    lg = CAP * np.tanh((rt.letters[: r["n"]] @ h) / CAP); rl = np.array(ref["softcapped_logits"])
    same += int(lg.argmax() == rl.argmax()); dl.append(float(np.abs(lg - rl).max()))
    print(f"{r['id']:28s} npu {np.round(lg, 2).tolist()}  ref {np.round(rl, 2).tolist()}  {ms:.0f} ms", flush=True)
print(f"same answer {same}/{len(dl)}; max|dlogit| median {np.median(dl):.2f} max {max(dl):.2f}")
