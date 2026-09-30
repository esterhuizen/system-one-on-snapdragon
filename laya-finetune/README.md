# laya-finetune: turn Laya into a dedicated classifier (Claude Opus labels -> fine-tune -> Snapdragon NPU)

The walkthrough, results and settings are in [../docs/LAYA-FINETUNE.md](../docs/LAYA-FINETUNE.md).

| File | What it does | Environment |
|---|---|---|
| `make_batches.py` | items.jsonl -> labelling batches + held-out split | any Python |
| `label_workflow.js` | Claude Code workflow: two independent Opus passes + adjudication, labels written to files | Claude Code ("use a workflow") |
| `finetune.py` | Laya encoder + one head per task, fine-tuned on the labels (prints the quick test as "epoch 0") | torch (CPU), laya, transformers |
| `export_npu.py` | static ONNX export + QNN graph fixes + CPU parity check | torch, laya, onnx, onnxscript, onnxruntime |
| `run_npu.py` | compile for the Hexagon NPU, predict with timing, optional scoring and CPU comparison | onnxruntime 1.30, onnxruntime-qnn 2.6.0, transformers |
| `examples/` | the helpdesk taxonomy (12 buckets, 6 work types) and labelling guide used in the write-up | |

Tested end to end on Windows 11 ARM64 (Snapdragon X Elite), Python 3.12 arm64.
