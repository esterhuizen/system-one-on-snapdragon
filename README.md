# System One models on Windows on Snapdragon

Running open "System One" typed-decision models on a **Snapdragon X Elite laptop** (Hexagon NPU, Adreno GPU, Oryon CPU), and
comparing them with TypeSafe's **Jev** API.

Typed-decision models answer structured questions about a piece of text with probabilities: pick one option, rate on a scale,
or yes/no. The ones covered here are:

- **[Laya](https://huggingface.co/convaiinnovations/laya)**: runs on the **NPU** and the **Adreno GPU**.
- **[Decider](https://github.com/Mapika/decider)**: runs fast on the **CPU** through llama.cpp.
- **[imajev](https://github.com/mohit67890/imajev)**: runs on the **CPU** through PyTorch.
- **[Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B)** (Gemma 4 12B): runs **entirely on the Hexagon NPU** at 4-bit, about
  2 s per request, through ONNX Runtime's QNN provider with no test-signing. It also runs on the CPU through its llama.cpp server.
  The build, the traps and the numbers are in **[docs/WINNOW-NPU.md](docs/WINNOW-NPU.md)**; the NPU-ready model files are on
  Hugging Face: [tielmane/Winnow-12B-NPU-LPBQ-X-Elite](https://huggingface.co/tielmane/Winnow-12B-NPU-LPBQ-X-Elite).
- **[decider-12b](https://huggingface.co/Mapika/decider-12b)** (Mapika, Gemma 4 12B): built for the **Hexagon NPU** with the same
  pipeline, about 2.2 s per request; JevBench standard 0.986, level with Jev. NPU files:
  [tielmane/decider-12b-NPU-LPBQ-X-Elite](https://huggingface.co/tielmane/decider-12b-NPU-LPBQ-X-Elite); code in `decider-npu/`.
- **Laya, fine-tuned into a dedicated classifier** with Claude Opus labels: on a real helpdesk-ticket job it went from 63.5% to
  **88.5%** (Jev 83.5%) and runs on the NPU at ~17 ms per ticket. Recipe and code: **[docs/LAYA-FINETUNE.md](docs/LAYA-FINETUNE.md)**.

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
| Winnow | Winnow-12B GGUF `b6ac22b0` (Q8_0); winnow-inference `6c2b3c0` (llama.cpp `911f6cdc`) built CPU-only with clang |
| Winnow on the NPU | build/quantize: onnxruntime-qnn 1.24.4; compile/run: onnxruntime 1.30.0 + onnxruntime-qnn 2.6.0 (QNN 2.39, HTP 2.50) |
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

**Winnow-12B** (requests of about 200–550 tokens, one or two questions):

| Path | Per request | Notes |
|---|---|---|
| **Hexagon NPU, LPBQ int4, questions packed into one pass** | **~2.2 s** | ~6 GB of NPU-mapped weights; up to 576 tokens per pass |
| Hexagon NPU, w8a16 | ~8 s | 11 GB of weights exceeds the NPU's ~9–10 GB mapping limit, which forces re-mapping |
| CPU, llama.cpp Q8_0 | ~6–8 s per short JevBench item | Needs about 14 GB free RAM; long option lists take several times longer |

**JevBench public items** (231; accuracy easy / standard / hard): Jev 1.000 / 0.986 / 0.712 · decider-2b 1.000 / 0.889 / 0.568 ·
imajev-2b 1.000 / 0.917 / 0.568 · Laya 0.958 / 0.694 / 0.351.
- **The 12B models on the NPU:** only 50 of the 111 hard items fit in 576 tokens, so they are scored on those 50.
  - decider-12b: 1.000 / **0.986** / 0.700.
  - Winnow-12B: 1.000 / 0.972 / 0.680.
  - The others on the same 50: Jev 0.760, decider-2b 0.680, CPU Winnow Q8_0 0.800.
- Details and caveats are in FINDINGS.

**Public tasks with human labels** (~3,900 items):

| Task | Metric | Jev | Winnow-12B NPU | decider-12b NPU | Laya NPU |
|---|---|---|---|---|---|
| Banking77 intent | accuracy | 0.930 | 0.868 | 0.870 | 0.823 |
| CFPB complaint product | accuracy | 0.824 | 0.760 | 0.776 | 0.581 |
| CFPB "debt not owed" | AUROC | 0.774 | 0.767 | 0.695 | 0.569 |
| Complaint tweets | AUROC | 0.969 | 0.948 | 0.877 | 0.827 |
| Complaint severity | weighted kappa | 0.669 | 0.609 | 0.666 | −0.030 |
| Enron frustration | AUROC | 0.945 | 0.922 | 0.758 | 0.769 |
| Enron politeness | Spearman | 0.595 | 0.598 | 0.411 | 0.280 |
| **Average of the seven** | mixed | **0.815** | 0.782 | 0.722 | 0.546 |
| **Average accuracy** (5 questions with one right answer) | accuracy | **0.857** | 0.823 | 0.821 | 0.647 |

decider-12b picks the right answer as often as Winnow. Its yes/no confidences, though, are almost always near 0 or 1:
Mapika's config sets a yes/no temperature of 0.05. Ranking metrics (AUROC) therefore come out lower. FINDINGS section 7 has
the details.

**A private helpdesk-ticket job** (200 held-out real tickets; two questions: bucket out of 12, work type out of 6; only
aggregates are published, details in [docs/LAYA-FINETUNE.md](docs/LAYA-FINETUNE.md)):

| Model | Bucket | Work type | Per ticket | Where it runs |
|---|---|---|---|---|
| **Laya fine-tuned on Claude Opus labels** | **88.5%** | **89.0%** | **0.017 s** | NPU |
| Jev 1.13.0 (TypeSafe API) | 83.5% | 85.5% | 0.2 s | cloud |
| Winnow-12B, Q8_0 | 78.5% | 81.0% | 29 s | CPU |
| decider-12b v2, 4-bit | 77.5% | 77.5% | 2.3 s | NPU |
| Winnow-12B, 4-bit, both questions in one pass | 75.5% | 73.5% | 2.1 s | NPU |
| Laya out of the box | 63.5% | 67.0% | 0.14 s | NPU |

The reference labels for those 200 tickets are a blind two-labeller Claude consensus, so the fine-tuned Laya (trained on
Claude Opus labels) may share some of their reading of the tickets.

Accuracy, the benchmark method, and every Windows ARM64 problem we hit are in **[docs/FINDINGS.md](docs/FINDINGS.md)**.

## Layout

```
laya/bin/      WSL launchers: winrun (run a Windows venv script), serve (start/stop Jev-compatible servers), run_suite
laya/win/      Windows-side Python: NPU/GPU servers, proofs, encoder export and QNN graph fixes, throttling opt-out
decider/       Windows ARM64 setup, CPU-only llama.cpp build, smoke tests, ONNX export + QNN diagnosis (bisect, chain)
imajev/        Windows ARM64 setup, torchvision-from-source build, smoke test
decider-npu/   decider-12b on the Hexagon NPU: Decider-format prompts/head on top of the winnow-npu pipeline, server, checks
laya-finetune/ Laya -> dedicated classifier: Opus labelling workflow, fine-tuning, NPU export and runner
winnow-npu/    Winnow-12B on the Hexagon NPU: GGUF -> ONNX chunks, calibration, LPBQ int4, packing, QNN compile, Jev-compatible
               server; tools/ (checks and probes); cpu/ (Winnow's llama.cpp server on the CPU: build, patch, launcher)
harness/       jevcmp.py (runner for any /v1/systemone endpoint), metrics and reports, benchmark item builders
docs/          FINDINGS.md, UPSTREAM.md, WINNOW-NPU.md and LAYA-FINETUNE.md (guides), fair-benchmark plan
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
