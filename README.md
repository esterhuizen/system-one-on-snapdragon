# System One models on Windows on Snapdragon

Running open "System One" typed-decision models on a **Snapdragon X Elite laptop** (Hexagon NPU, Adreno GPU, Oryon CPU), and
comparing them with TypeSafe's **Jev** API.

Typed-decision models answer structured questions about a piece of text with probabilities: pick one option, rate on a scale,
or yes/no. The ones covered here are:

- **[Laya](https://huggingface.co/convaiinnovations/laya)**: runs on the **NPU** and the **Adreno GPU**.
- **[Decider](https://github.com/Mapika/decider)**: runs fast on the **CPU** through llama.cpp.
- **[imajev](https://github.com/mohit67890/imajev)**: runs on the **CPU** through PyTorch.

**This repository is a dated snapshot, not a maintained fork.** Each model lives in its upstream repository, and the fixes that
belong there are proposed there ([docs/UPSTREAM.md](docs/UPSTREAM.md)). What this repo keeps is the glue: setup scripts,
Jev-compatible servers, proof scripts, and a benchmark harness. Versions, model revisions and hashes are pinned, so the scripts
stay reproducible even as upstream moves on.

## Tested on (September 2026)

| Component | Version |
|---|---|
| Hardware | Snapdragon X Elite X1E80100 (Hexagon HTP v73), Adreno X1-85 GPU (driver 31.0.133.1), 32 GB RAM |
| OS | Windows 11 26200 (ARM64), with WSL2 Ubuntu 24.04 (aarch64) used for orchestration |
| Python | Native ARM64 CPython 3.12. Avoid x64 Python, which runs emulated. |
| Laya | laya 0.3.20; piffie/laya-snapdragon v0.2.0; onnxruntime-qnn 1.24.4 (NPU) and 2.6.0 with onnxruntime 1.30.0 (GPU) |
| Decider | decider-ai 1.5.0; decider-2b v11 (HF `533964da…`, GGUF `ffa92e6d…`); llama-cpp-python 0.3.35 built CPU-only with clang |
| imajev | imajev 1.0.0; adapter `531ea011…`; Qwen3.5-2B `15852e8c…`; torchvision 0.29.0 built from source |
| PyTorch | 2.14.0+cpu, from download.pytorch.org (PyPI has no Windows ARM64 torch) |

## Results in one table

Median time for one request with 3–4 short questions (about 100–130 tokens):

| Model and path | Latency | Notes |
|---|---|---|
| Laya on the NPU (QNN HTP, fp16) | **~134 ms** | ~33 ms per question at 128 tokens; answers match CPU on 99.8% |
| Laya on the Adreno GPU (QNN GPU, fp32) | ~590 ms | Answers bit-identical to CPU |
| Decider-2b, llama.cpp CPU-only, Q8_0 | **~220 ms** | 50× faster than PyTorch on the same CPU |
| Decider-2b, PyTorch CPU, float32 | ~11 s | bfloat16 is 13× slower still on this CPU |
| imajev-2b, PyTorch CPU, float32 | ~14.5 s | Needs ≥10.5 GB free RAM |
| Jev API (reference) | ~220 ms | Includes ~160 ms network round trip |

Accuracy, the benchmark method, and every Windows ARM64 problem we hit are in **[docs/FINDINGS.md](docs/FINDINGS.md)**.

## Layout

```
laya/bin/      WSL launchers: winrun (run a Windows venv script), serve (start/stop Jev-compatible servers), run_suite
laya/win/      Windows-side Python: NPU/GPU servers, proofs, encoder export and QNN graph fixes, throttling opt-out
decider/       Windows ARM64 setup, CPU-only llama.cpp build, smoke tests, ONNX export + QNN diagnosis (bisect, chain)
imajev/        Windows ARM64 setup, torchvision-from-source build, smoke test
harness/       jevcmp.py (runner for any /v1/systemone endpoint), metrics and reports, benchmark item builders
docs/          FINDINGS.md, UPSTREAM.md, fair-benchmark plan
```

## Configuration

- **Windows side:** each model uses its own working folder under `%USERPROFILE%` (`local-laya`, `local-decider`,
  `local-imajev`), each containing `envs\`, `models\`, `results\` and `scripts\`.
- **WSL side**, before using `laya/bin/*`:
  ```bash
  export WIN_HOME=/mnt/c/Users/<you>
  export WINUSER=<you>
  ```
- **uv:** expected on the Windows `PATH`.
- **Jev key:** the harness reads `TYPESAFE_API_KEY` from the environment only. Never commit it.

## Quick start

**Laya on the NPU**
1. Run `laya/win/setup_env.ps1 -Name npu`.
2. Build the NPU contexts:
   ```bash
   laya/bin/winrun npu -m laya_snapdragon --models <models\npu> build --seq 128 256 512
   ```
3. Prove it runs on the NPU: `laya/bin/winrun npu npu_proof.py --seq 128`.
4. Serve it: `laya/bin/serve start npu`.

**Decider (fast CPU path)**
1. Run `decider/setup_cpu.ps1`.
2. Run `decider/build_llamacpp.cmd`. It needs the Visual Studio Build Tools "C++ Clang tools for Windows" component, and builds with every GPU backend off.
3. Download the GGUF at the pinned revision.
4. Run `decider/smoke_gguf.py`.

**imajev**
1. Run `imajev/setup_cpu.ps1`.
2. Run `imajev/build_torchvision.cmd`.
3. Download the pinned base model and adapter.
4. Run `imajev/smoke.py`.

**Comparisons**
1. Build items with `harness/fairbench/*` (public human-labelled data) or use `harness/items/*.jsonl`.
2. Run `laya/bin/run_suite <items> <run>`.
3. Produce the report with `harness/report.py` or `harness/fair_report.py`.

## Safety notes

- **Never run llama.cpp's Adreno OpenCL GPU backend on this setup.** It coincided with two blue screens (bugcheck 0xD1) during
  testing. The build here compiles every GPU backend out. See FINDINGS.
- **Servers bind to 127.0.0.1 and have no authentication.** Don't expose them.
- **Check free RAM before loading a model.** The smoke tests refuse to start below a set threshold.

## About the test data

`harness/items/suite-v1.jsonl` is a synthetic suite written by Claude models; nothing in it is real customer data. For publication,
its invented email addresses, URLs, IP addresses and phone numbers were rewritten to reserved example forms (`*.example` domains,
the 203.0.113.0/24 documentation range, 5-filled numbers) **after** the benchmark runs. A few items therefore differ slightly from
the text the reported numbers were measured on. The public-benchmark items are not included; `harness/fairbench/` rebuilds them
from their original sources.

## Licence

Apache-2.0 for the code in this repository. Models and datasets keep their own licences; this repo contains none of their
weights or data, only scripts that fetch them at pinned revisions.
