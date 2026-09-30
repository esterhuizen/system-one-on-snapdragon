"""Export a fine-tuned classifier (finetune.py output) to a static [1, max_len] ONNX graph for the Hexagon NPU:
encoder + mean pool + one softmax output per task (p_<task>). Applies the two graph fixes used for Laya's encoder
(../laya/win/fix_allowzero.py, ../laya/win/qnn_fix.py) and checks ONNX Runtime CPU against PyTorch.
Needs torch, laya, onnx, onnxscript and onnxruntime in one environment.

python export_npu.py --ckpt-dir OUT --out OUT/onnx/classifier.onnx      -> also writes OUT/onnx/classifier_q.onnx (use this one)
"""
import argparse, json, os, subprocess, sys
import numpy as np
import torch
import torch.nn as nn
import laya
from safetensors.torch import load_file
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt-dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
cfg = json.load(open(os.path.join(a.ckpt_dir, "classifier_config.json"))); TASKS = cfg["tasks"]; S = cfg["max_len"]


class Classifier(nn.Module):
    def __init__(self, enc):
        super().__init__(); self.enc = enc; d = enc.config.hidden_size; self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({t: nn.Linear(d, len(k)) for t, k in TASKS.items()})

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask[..., None].to(h.dtype); p = (h * m).sum(1) / m.sum(1)
        return tuple(self.heads[t](p).softmax(-1) for t in TASKS)


model = Classifier(laya.load(cfg["snapshot"]).model.encoder)
model.load_state_dict(load_file(os.path.join(a.ckpt_dir, "classifier.safetensors")), strict=True); model.eval().float()
tok = AutoTokenizer.from_pretrained(os.path.join(cfg["snapshot"], "tokenizer"))
b = tok(["An example request\nType: Incident\nCategory: Email"], padding="max_length", truncation=True, max_length=S, return_tensors="pt")
with torch.inference_mode(): ref = [t.numpy() for t in model(b["input_ids"], b["attention_mask"])]
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
prog = torch.onnx.export(model, (b["input_ids"], b["attention_mask"]), input_names=["input_ids", "attention_mask"],
                         output_names=[f"p_{t}" for t in TASKS], opset_version=18, dynamo=True, external_data=True)
prog.optimize(); prog.save(a.out, external_data=True)
import onnxruntime as ort
out = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"]).run(None, {"input_ids": b["input_ids"].numpy(), "attention_mask": b["attention_mask"].numpy()})
print("ORT CPU vs torch max|dp|:", {t: float(np.abs(o - r).max()) for t, o, r in zip(TASKS, out, ref)}, flush=True)
az, q = a.out.replace(".onnx", "_az.onnx"), a.out.replace(".onnx", "_q.onnx")
win = os.path.join(HERE, "..", "laya", "win")
subprocess.run([sys.executable, os.path.join(win, "fix_allowzero.py"), a.out, az], check=True)
subprocess.run([sys.executable, os.path.join(win, "qnn_fix.py"), az, q], check=True)
print("NPU-ready graph:", q)
