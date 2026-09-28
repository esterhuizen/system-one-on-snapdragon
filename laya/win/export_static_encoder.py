"""Export Laya's encoder (ModernBERT-large) with STATIC shapes [batch, seq] (needed by QNN; avoids DML int64 shape ops).

python export_static_encoder.py --out-dir DIR --batch 1 --seqs 128,512 [--fp16]
"""
import argparse, json, os
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from laya.common import build_model


class EncoderOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seqs", default="128,512")
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()
    prefix = f"{a.subfolder}/" if a.subfolder else ""
    root = snapshot_download(a.repo, allow_patterns=[prefix + p for p in ("rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*")])
    d = os.path.join(root, a.subfolder) if a.subfolder else root
    cfg = json.load(open(os.path.join(d, "rl_agent_config.json")))
    model = build_model(cfg, encoder_dir=os.path.join(d, "encoder"), pretrained=False)
    model.load_state_dict(load_file(os.path.join(d, "model.safetensors")), strict=False)
    model.eval().float()
    enc = EncoderOnly(model).eval()
    os.makedirs(a.out_dir, exist_ok=True)
    for seq in [int(s) for s in a.seqs.split(",")]:
        ids = torch.randint(1000, 50000, (a.batch, seq), dtype=torch.long)
        mask = torch.ones(a.batch, seq, dtype=torch.long)
        with torch.inference_mode():
            ref = enc(ids, mask)
        for dt in (["fp32", "fp16"] if a.fp16 else ["fp32"]):
            path = os.path.join(a.out_dir, f"encoder_b{a.batch}_s{seq}_{dt}.onnx")
            m = enc if dt == "fp32" else EncoderOnly(model).half().eval()
            prog = torch.onnx.export(m, (ids, mask), input_names=["input_ids", "attention_mask"],
                                     output_names=["last_hidden_state"], opset_version=18, dynamo=True,
                                     external_data=True)
            prog.optimize()
            prog.save(path, external_data=True)
            import onnxruntime as ort, numpy as np
            s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            out = s.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0].astype(np.float32)
            diff = float(np.abs(out - ref.numpy()).max())
            print(f"wrote {path}  CPU-ORT vs torch fp32 max|d|={diff:.3g}  ops={sorted({n.op_type for n in prog.model_proto.graph.node}) if hasattr(prog,'model_proto') else ''}", flush=True)
            if dt == "fp16":
                model.float()


if __name__ == "__main__":
    main()
