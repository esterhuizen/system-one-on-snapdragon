# Findings

Measured on one Snapdragon X Elite laptop in September 2026 (versions in the README). The Jev figures come from the TypeSafe API
(`jev-1.13.0`).

## 1. Laya on the NPU and GPU

- **NPU route:** piffie/laya-snapdragon v0.2.0 works on the **X Elite (Hexagon HTP v73)**. It had only been reported on the X2
  Elite. The fp16 contexts compiled in about 1 minute per size, and the 512-token size in about 90 s.
- **Proof it runs on the NPU:**
  - Each compiled graph is EPContext-only, so it cannot run on the CPU.
  - Placement is QNN for everything except one GatherND on the CPU.
  - Answers were identical to the CPU model on 12 of 12 verification cases, with a maximum probability difference of 0.004.

| Tokens per row | NPU (fp16) | Adreno GPU (QNN GPU, fp32) | ORT CPU |
|---|---|---|---|
| 128 | **33 ms** | 83 ms | 143–175 ms |
| 256 | **100 ms** | — | 424 ms |
| 512 | 385 ms | **317 ms** | 751–1083 ms |

- **Crossover:** the NPU wins below about 256 tokens. At 512 tokens it spills about 4 GB per pass out of its fast on-chip memory
  (VTCM), and the GPU pulls ahead.
- **GPU route:** we exported Laya's encoder at fixed sizes (fp32), applied two graph fixes, and ran it on QNN's GPU backend.
  - Fix 1: `Reshape allowzero=1→0`.
  - Fix 2: remove SDPA's `Where(IsNaN(x),0,x)` guard. Without it, finalization fails with error 6022. The removal is exact here
    because the mask uses finfo.min, not -inf.
  - Answers were bit-identical to torch CPU.
- **Per-question cost:** one question per NPU pass, so latency grows linearly with the number of questions. Jev evaluates all
  questions together.

## 2. Accuracy: Laya vs Jev

### Harness checks against published numbers

| Benchmark | Jev here | Jev published | Laya here | Laya published |
|---|---|---|---|---|
| sysone-bench Banking77 (96) | 89.6% | 90.6% | 80.2% | 80.2% |
| sysone-bench SST-5 (60, within 0.5 level) | 60.0% | 61.7% | 36.7% | 36.7% |
| LocalLLaMA/typed-decisions test (2,000) | 73.3% | 72.7% | 36.2% | 36.2% |

### Public tasks with human-written answers (~9,500 decisions; Jev ×3 runs, Laya on the NPU)

| Task | Metric | Jev | Laya |
|---|---|---|---|
| Banking77 intent (12 options) | accuracy | 0.930 | 0.823 |
| CFPB complaint product (7 options) | accuracy | 0.824 | 0.581 |
| CFPB "consumer says debt not owed" | AUROC | 0.774 | 0.569 |
| Complaint tweets: is it a complaint? | AUROC | 0.969 | 0.827 |
| Complaint severity (4 levels) | weighted kappa | 0.669 | −0.030 |
| Enron email: writer frustrated? | AUROC | 0.945 | 0.769 |
| Enron email politeness | Spearman | 0.595 | 0.280 |

- **Laya zero-shot:** usable for short intent routing, weak on judgement. Its yes/no answers are badly biased.
- **Alternative wording:** asking the yes/no questions as a two-option A/B choice (Laya's README workaround) changes balanced
  accuracy by only a few points.
- **Fresh data:** Jev's lead holds on CFPB complaints from March–August 2026.
- **Item builders:** `harness/fairbench/` rebuilds the items from their public sources. The data is not redistributed.

### Claude-written synthetic suite (128 tickets and emails, 543 questions; `harness/items/suite-v1.jsonl`)

| | 128-token items | 512-token items |
|---|---|---|
| Jev | 98.8% | 99.0% |
| Laya | 61.5% | 47.9% |

- **Caveat:** the correct answers are Claude-consensus labels, which favour LLM-like models.
- **Format isn't the gap:** rewriting the questions the way Laya's docs recommend flipped about 14% of Laya's answers, but the
  total stayed the same.

## 3. Decider and imajev on this laptop

- **Decider, numpy:** `decider-ai` pins `numpy<2`, and numpy 1.x has no Windows ARM64 wheel. Install decider with `--no-deps` on
  top of numpy 2. Its only runtime numpy use works on 2.x.
- **bfloat16 is about 13× slower than float32** on Windows ARM64 PyTorch (Decider-2b: 137 s vs 11 s per request).
  Pass `dtype=torch.float32` on CPU.
- **llama.cpp needs clang:** it refuses to build with MSVC on ARM64 ("MSVC is not supported for ARM, use clang"). Use the Build
  Tools "C++ Clang tools for Windows" component. Build with `-DGGML_OPENMP=OFF` and **every GPU backend off**.
  - Result: Decider-2b Q8_0 at about 220 ms per 3-question request (Q4_K_M about 330 ms; Q8_0 is faster on this CPU).
- **imajev needs torchvision even for text:** transformers' Qwen3.5 processor loads a video processor that imports it. There's no
  Windows ARM64 torchvision 0.29 wheel, so `imajev/build_torchvision.cmd` compiles it (MSVC ARM64, CPU-only, codecs off).
- **imajev memory:** its CPU path loads in float32, which needs about 9.5 GB for the 2B (about 18 GB for the 4B).

## 4. Decider on the Adreno GPU via QNN: blocked by a per-process graph-size limit

- **Export:** Decider-2b exports to a static ONNX graph (S=256, 8 answer slots, fp16) as backbone plus option-letter readout. The
  export took 52 minutes: Qwen3.5's Gated DeltaNet layers trace as unrolled loops, giving about 20,000 ops, including about 2,350
  Scatter ops.
- **Parity:** ONNX Runtime on the CPU matches the GGUF answers (0.974 / 0.988 / 0.878 vs 0.974 / 0.988 / 0.865).
- **Result:** QNN GPU finalization of the full graph fails with error 6022. Diagnosis (`decider/qnn_bisect.py`,
  `decider/qnn_chain.py`):

| Test | Result |
|---|---|
| Each ~500-node chunk alone | All 40 compile (3-10 s) |
| One graph of up to ~3,000 nodes | Compiles |
| One graph of ~3,500+ nodes, anywhere in the model | Fails, 6022 |
| 1,000-node segments chained in one process | 3 compile, the 4th fails |
| That 4th segment alone in a fresh process | Compiles |

- **Conclusion:** no operation is unsupported. onnxruntime-qnn 2.6.0's GPU backend holds only about 3,000 nodes of compiled
  graphs **per process** (cumulative across sessions). Laya's encoder (~1,800 nodes) fits; an unrolled Qwen3.5 DeltaNet model is
  about 7× over. Even the segments that compiled put 12-15% of nodes on the CPU.
- **Why we stopped:** splitting across 7+ processes might work, but the hand-offs and CPU fallback make it unlikely to beat the
  CPU-only llama.cpp path (~220 ms). That remains the recommended way to run Decider on this laptop.

## 5. Windows ARM64 operational problems

- **Launching Windows servers from WSL hangs WSL** when you use PowerShell `Start-Process`, because the QNN driver keeps WSL's
  interop console relay open. Launch through WMI `Win32_Process.Create` instead (`laya/bin/serve`).
- **Hidden background servers are throttled** by Windows efficiency mode (EcoQoS). The CPU paths ran about 4× slower. Opt out in
  the process (`laya/win/nothrottle.py`).
- **WSL can't reach Windows `127.0.0.1`** in NAT networking mode, so run HTTP clients on the Windows side.
- **openpyxl 3.1.2 files fail a plain open in Excel** (COM "Unable to get the Open property"). openpyxl 3.1.5 on Windows works.

## 6. Blue screens during llama.cpp GPU experiments

Two **0xD1 DRIVER_IRQL_NOT_LESS_OR_EQUAL** bugchecks occurred while prebuilt llama.cpp Windows ARM64 binaries were being tried
with the **Adreno OpenCL** backend (build b10453) and with every backend loaded.

The faulting driver has not been confirmed: the minidumps need admin access to read. Everything else stayed stable, including
QNN's GPU and NPU paths and CPU-only llama.cpp. Until the driver is identified, avoid llama.cpp's OpenCL backend on this
driver version (31.0.133.1).
