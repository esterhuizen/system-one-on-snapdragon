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

- **Decider, numpy** (*fixed in decider-ai 1.7.1*): `decider-ai` up to 1.6 pinned `numpy<2`, and numpy 1.x has no Windows ARM64
  wheel, so we installed it with `--no-deps` on top of numpy 2. Since 1.7.1 the pin is gone: a fresh venv installs 1.8.1 on
  numpy 2.5.3, with torch from the PyTorch CPU index.
- **bfloat16 is about 13× slower than float32** on Windows ARM64 PyTorch (Decider-2b: 137 s vs 11 s per request).
  *Fixed in decider-ai 1.7.1:* the CPU default is now float32. With 1.8.1 the same 3-question request takes 5.2–5.4 s.
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

## 5. JevBench public items: every model on this laptop

The unmodified [JevBench](https://github.com/fstandhartinger/jevbench) harness (`9ec6f15`) ran on Windows ARM64. It needed a
no-op `fcntl` shim: its budget ledger imports the Unix-only module, and a single-process run needs no locking. Each local
system was served as a Jev-compatible `/v1/systemone` endpoint and driven through the harness's own `typesafe` adapter, one
system at a time. Laya also ran through the harness's in-process `laya_local` adapter.

| System (231 public items) | Runs on | Easy (48) | Standard (72) | Hard (111) | Hard ECE | Median, short / hard |
|---|---|---|---|---|---|---|
| Jev 1.13.0 | cloud API | 1.000 | 0.986 | 0.712 | 0.083 | 256 / 258 ms |
| decider-2b v11, GGUF Q8_0 | CPU, llama.cpp | 1.000 | 0.889 | 0.568 | 0.184 | 196 ms / 1.8 s |
| imajev-2b (1 rotation + calibration) | CPU, PyTorch fp32 | 1.000 | 0.917 | 0.568 | 0.096 | 6.4 / 22 s |
| Laya (ModernBERT-large) | NPU fp16 | 0.958 | 0.694 | 0.351 | 0.198 | 40 / 412 ms |
| Laya | Adreno GPU / CPU | 0.958 | 0.694 | 0.351 | 0.19 | 130 ms / 736 ms (short) |
| Winnow-12B, LPBQ int4 (section 7) | NPU | 1.000 | 0.972 | see below | | 2.2 s |

- **The harness reproduces published results:**
  - Laya matches JevBench's published per-item outcomes on **231/231** items, on all three routes.
  - Jev matches on 229/231.
  - decider-2b GGUF is within one hard item of Mapika's own v11 figure (0.568 vs 0.577).
- **imajev:** one very long hard item exceeded the harness's 120 s timeout on the CPU and counts as wrong. The author's 0.604 on
  hard used 4 rotations.
- **Winnow on the NPU, hard tier:** 61 of the 111 hard items are longer than one 576-token pass and are refused. On the 50 that
  fit, the scores are Winnow NPU **0.680**, Jev 0.760, decider-2b 0.680, imajev-2b 0.660 and Laya 0.300. On easy and standard
  items the NPU build gives the same answer as CPU Winnow on 64 of 67 overlapping items.
- **The 4-bit cost, measured:** CPU Winnow (Q8_0) scores **0.800** on the same 50 hard items (median 17.5 s per item), ahead of
  Jev's 0.760. So full-precision Winnow is strong on reasoning, and the 4-bit NPU build loses 12 points (6 items) there.
  CPU and NPU gave the same answer on 41 of 50.
- *Correction (2026-09-30): an earlier version of this page gave Winnow on the NPU 0.875 (standard) and 0.400 (hard). That run went through a multi-threaded server, which silently corrupts NPU results after about 60–120 requests (see the traps in [WINNOW-NPU.md](WINNOW-NPU.md)). The numbers here come from a re-run on the fixed single-threaded server.*
- **What the hard tier separates:** long policy texts, multi-hop and temporal-numeric questions. Easy items are nearly solved
  by every model.

## 6. Winnow-12B on the CPU

- **Build:** Winnow's server (winnow-inference `6c2b3c0`, a patched llama.cpp `911f6cdc`) builds for Windows ARM64 with the
  Visual Studio clang, CPU only, with every GPU backend off. Its own unit tests pass.
- **Two fixes** to run it without a GPU:
  - The decision engine refused to start ("requires a GPU backend"). Its device handle is only used for memory figures, so a
    two-line fallback to the CPU device (`winnow-npu/cpu/local-cpu-fallback.patch`) is enough.
  - The first request crashed (access violation 0xc0000005). The answer head reads the tied embedding rows back with
    `ggml_backend_tensor_get`, which llama.cpp's CPU repack buffer does not implement. Start with **`--no-repack`**; pinning the
    embedding to a plain CPU buffer did not help.
- **Memory:** `--load-mode none` (this llama.cpp has no `--no-mmap`) avoids holding the weights twice. Q8_0 then needs about
  13.6 GB resident, against 19 GB with mmap plus repack.
- **Speed:** about 2 s for a short 2-question request and 6–8 s per JevBench easy item. Requests with long option lists take
  several times longer. It is usable for checking, not for volume.

## 7. Winnow-12B on the Hexagon NPU

The full guide is in **[WINNOW-NPU.md](WINNOW-NPU.md)**. In short:

- **Route:** ONNX Runtime's QNN execution provider. That means no llama.cpp Hexagon backend, which on Windows would need
  test-signing and therefore Secure Boot off.
- **Graphs:** the GGUF weights become 12 static ONNX graphs of 4 decoder layers. Each is calibrated (uint16 activations),
  converted to **LPBQ int4** weights (block 32) and compiled to a QNN context.
- **Head:** the embeddings and the answer head run on the CPU.
- **Checks:**
  - a PyTorch rebuild from the GGUF matches the CPU server;
  - the Python port of the prompt and tokenizer matches the server's token ids on 229/229 requests;
  - a single layer in fp16 on the NPU matches PyTorch to 8e-4.
- **Speed path:**
  - CPU Q8_0: 6–8 s per JevBench easy item;
  - NPU w8a16: 4.2 s per 512-token pass;
  - NPU LPBQ int4: 1.6 s per 512-token pass;
  - all of a request's questions packed into one 576-token pass: **2.2 s per JevBench item**.
- **The limits we found:**
  - the NPU keeps only about **9–10 GB** of weights mapped, system-wide, so the 11 GB w8a16 chain re-maps on every pass;
  - LPBQ block scales must be **4-bit integers (1..15)**;
  - onnxruntime-qnn **1.24.4 rejects LPBQ**, while the 2.6.0 plugin runs it;
  - `SimplifiedLayerNormalization` falls back to the CPU, but opset 23 `RMSNormalization` does not.
- **Accuracy cost:**
  - one layer at LPBQ int4 has about 12% relative output error (w8a16: about 2%);
  - end to end, the NPU answer matched CPU Winnow on 64/67 JevBench easy and standard items;
  - the loss shows on close calls and on multi-step reasoning: on the hard JevBench items that fit, CPU Q8_0 scores 0.800 and
    the 4-bit NPU build 0.680 (section 5). On classification-style public tasks it stays close to Jev (section 7).

### Winnow-12B and decider-12b on the NPU against Jev and Laya: public tasks with human labels

| Public task (human labels; ~3,900 items) | Metric | Jev 1.13.0 | **Winnow-12B, NPU, LPBQ int4** | **decider-12b v2, NPU, LPBQ int4** | Laya, NPU |
|---|---|---|---|---|---|
| Banking77 intent (12 options) | accuracy | 0.930 | 0.868 | **0.870** | 0.823 |
| CFPB complaint product (7 options) | accuracy | 0.824 | 0.760 | **0.776** | 0.581 |
| CFPB "consumer says debt not owed" | AUROC | 0.774 | **0.767** | 0.695 | 0.569 |
| Complaint tweets: is it a complaint? | AUROC | 0.969 | **0.948** | 0.877 | 0.827 |
| Complaint severity (4 levels) | weighted kappa | 0.669 | 0.609 | **0.666** | −0.030 |
| Enron email: writer frustrated? | AUROC | 0.945 | **0.922** | 0.758 | 0.769 |
| Enron email politeness | Spearman | 0.595 | **0.598** | 0.411 | 0.280 |
| **Average of the seven rows above** | mixed | **0.815** | 0.782 | 0.722 | 0.546 |
| **Average accuracy** (Banking77, CFPB product, debt not owed, complaint, frustrated) | accuracy | **0.857** | 0.823 | 0.821 | 0.647 |

- **How to read the averages:**
  - The first average mixes accuracy, AUROC, kappa and Spearman. That makes it a rough overall score, not a single
    metric.
  - The second averages plain accuracy on the five questions that have a single right answer. It asks only "did it
    pick the right answer?", leaving out how good its confidence is.
- **decider-12b's yes/no confidences are all-or-nothing, by design.**
  - Mapika's `decider_config.json` for 12b-v2 sets the yes/no temperature to 0.05. 98.5% of its yes/no answers come out
    below 0.01 or above 0.99, so AUROC (which ranks items by confidence) loses most of its information.
  - Its yes/no accuracy is close to Winnow's: complaint 0.874 against 0.873, frustrated 0.903 against 0.910.
  - Asked as an A/B choice (the suite's twin questions, choice temperature 1.5), its AUROC is 0.942 (tweets), 0.934
    (Enron) and 0.775 (CFPB), in line with Winnow's 0.945, 0.912 and 0.753.
  - **The cause is the temperature alone.**
    - We re-ran all 2,975 items with yes/no questions, keeping each answer's raw [no, yes] letter logits
      (`DECIDER_NPU_RAW=1`), and re-scored them at other temperatures (`harness/noul_temperature.py`).
    - At T = 0.05 the re-scoring reproduces the served results exactly.
    - At T = 1.0 the outputs are no longer rounded to exactly 0 or 1, and AUROC recovers fully: tweets 0.943, Enron 0.934,
      CFPB 0.754 (Winnow 0.948 / 0.922 / 0.767). Accuracy is unchanged.
    - Calibration is best at about T = 1.5–2.0:
      - the best single value across the three tasks, by log loss, is 1.97;
      - fitting on two tasks and testing on the third gives 1.6–2.8;
      - ECE is 0.025 (tweets) and 0.033 (Enron), against 0.12 and 0.10 at T = 0.05.

    | Yes/no AUROC | T = 0.05 (as shipped) | T = 1.0 | T = 1.5 |
    |---|---|---|---|
    | Complaint tweets | 0.877 | 0.943 | 0.943 |
    | Enron: writer frustrated? | 0.758 | 0.934 | 0.934 |
    | CFPB "debt not owed" | 0.695 | 0.754 | 0.755 |

    With T = 1.5 for yes/no, decider-12b's seven-row average is **0.765** (Winnow 0.782); the remaining gap is mostly
    politeness. The table above keeps the as-shipped numbers. Measured at 4-bit on the NPU; reported to Mapika in
    [decider#18](https://github.com/Mapika/decider/issues/18).
  - Until the config changes, use decider-12b's yes/no answers as decisions rather than as scores to rank or threshold,
    or set `temperature_by_type.noul` to about 1.5 yourself.
- **Politeness** (a 5-level rating) is decider-12b's one clear weak spot here.
- **Same items and the same harness** (`harness/jevcmp.py`, `fair_report.py`) as section 2. Jev ran 3 repeats; Laya,
  Winnow and decider-12b ran 1 each.
- **Items refused by the NPU servers:** an item whose state plus question is longer than 576 tokens is refused, and is left
  out of that model's rows. Winnow refused 87 CFPB and Enron items; decider-12b, with its shorter prompt, refused 18.
- **Speed:** median 2.1–2.2 s per item for both 12B models, whether an item has 1, 2 or 3 questions, since they are packed
  into one pass.
- **Recovery:** the Winnow NPU server recovered by itself once, when the NPU subsystem restarted mid-run.
- **Discarded runs:** two earlier Winnow runs went through a multi-threaded server and returned chance-level answers. They
  are discarded; see the traps in WINNOW-NPU.md.

## 8. Windows ARM64 operational problems

- **Launching Windows servers from WSL hangs WSL** when you use PowerShell `Start-Process`, because the QNN driver keeps WSL's
  interop console relay open. Launch through WMI `Win32_Process.Create` instead (`laya/bin/serve`).
- **Hidden background servers are throttled** by Windows efficiency mode (EcoQoS). The CPU paths ran about 4× slower. Opt out in
  the process (`laya/win/nothrottle.py`).
- **WSL can't reach Windows `127.0.0.1`** in NAT networking mode, so run HTTP clients on the Windows side.
- **openpyxl 3.1.2 files fail a plain open in Excel** (COM "Unable to get the Open property"). openpyxl 3.1.5 on Windows works.

## 9. Blue screens during llama.cpp GPU experiments

Two **0xD1 DRIVER_IRQL_NOT_LESS_OR_EQUAL** bugchecks occurred while prebuilt llama.cpp Windows ARM64 binaries were being tried
with the **Adreno OpenCL** backend (build b10453) and with every backend loaded.

The faulting driver has not been confirmed: the minidumps need admin access to read. Everything else stayed stable, including
QNN's GPU and NPU paths and CPU-only llama.cpp. Until the driver is identified, avoid llama.cpp's OpenCL backend on this
driver version (31.0.133.1).
