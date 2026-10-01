# Running a 12B decision model on the Snapdragon X Elite NPU

This guide shows how to run **Winnow-12B** entirely on the **Hexagon NPU** of a Snapdragon X Elite laptop, and what it took to
get there. Winnow-12B is a Gemma 4 12B fine-tune that answers typed questions (choice, yes/no, score) with probabilities, in
the same wire format as TypeSafe's Jev.

The route uses ONNX Runtime's QNN execution provider, so it needs no test-signing and no Secure Boot changes. The result is a
local `/v1/systemone` server that answers a short request in about **2 seconds** from about **6 GB** of NPU-mapped 4-bit
weights.

Everything here was measured on **one** laptop (Snapdragon X Elite X1E80100, Hexagon HTP v73, 32 GB RAM, Windows 11 26200) in
late September 2026. Treat the numbers as a dated snapshot, not a promise.

**Ready-made model files** (the 12 publicly calibrated 4-bit chunks, plus compiled X Elite contexts):
[tielmane/Winnow-12B-NPU-LPBQ-X-Elite](https://huggingface.co/tielmane/Winnow-12B-NPU-LPBQ-X-Elite) on Hugging Face. With those
you can skip steps 5–7 below.

## Contents

- [Results](#results)
- [How it works](#how-it-works)
- [Step by step](#step-by-step)
- [The traps, and what fixed each one](#the-traps-and-what-fixed-each-one)
- [From 7 s to 2 s](#from-7-s-to-2-s)
- [Checking your build](#checking-your-build)
- [decider-12b: the same recipe](#decider-12b-the-same-recipe)
- [Limits](#limits)
- [Credits and licences](#credits-and-licences)

## Results

**JevBench public items** (231; [JevBench](https://github.com/fstandhartinger/jevbench) harness, unmodified, same items for
every system):

| System | Easy (48) | Standard (72) | Hard items that fit in 576 tokens (50 of 111) | Median time per item |
|---|---|---|---|---|
| **Winnow-12B, NPU, LPBQ int4 (this guide)** | **1.000** | **0.972** | **0.680** | **2.2 s** |
| Jev 1.13.0 (TypeSafe API) | 1.000 | 0.986 | 0.760 | 0.26 s |
| decider-2b v11, CPU (llama.cpp Q8_0) | 1.000 | 0.889 | 0.680 | 0.2 s short / 1.8 s long |
| imajev-2b, CPU (PyTorch fp32) | 1.000 | 0.917 | 0.660 | 6–22 s |
| Laya, NPU (fp16) | 0.958 | 0.694 | 0.300 | 0.04–0.4 s |

- **Easy and standard:** the NPU build keeps Winnow's classification accuracy. It gave the same answer as CPU Winnow (Q8_0) on
  64 of 67 overlapping items.
- **Hard:** 61 of the 111 hard items (long policy texts) exceed one 576-token pass, and the server refuses them. On the 50 that
  fit, the 4-bit build scores 0.680: level with decider-2b, behind Jev's 0.760. Full-precision Winnow on the CPU (Q8_0) scores
  **0.800** on the same 50 items, so 4-bit rounding costs about 12 points on multi-step reasoning. On classification-style
  tasks the cost is small (see the next table).

*Correction (2026-09-30): an earlier version of this page gave Winnow on the NPU 0.875 (standard) and 0.400 (hard). That run went through a multi-threaded server, which silently corrupts NPU results after about 60–120 requests (see the traps below). The numbers here come from a re-run on the fixed single-threaded server.*

**Public tasks with human labels** (the fair suite from this repo's `harness/fairbench`, about 3,900 items, the same items for every
system):

| Public task (human labels; ~3,900 items) | Metric | Jev 1.13.0 | **Winnow-12B, NPU, LPBQ int4** | Laya, NPU |
|---|---|---|---|---|
| Banking77 intent (12 options) | accuracy | 0.930 | **0.868** | 0.823 |
| CFPB complaint product (7 options) | accuracy | 0.824 | **0.760** | 0.581 |
| CFPB "consumer says debt not owed" | AUROC | 0.774 | **0.767** | 0.569 |
| Complaint tweets: is it a complaint? | AUROC | 0.969 | **0.948** | 0.827 |
| Complaint severity (4 levels) | weighted kappa | 0.669 | **0.609** | −0.030 |
| Enron email: writer frustrated? | AUROC | 0.945 | **0.922** | 0.769 |
| Enron email politeness | Spearman | 0.595 | **0.598** | 0.280 |

- On these classification and judgement tasks, the 4-bit NPU build lands **close to Jev** (within about 1–7 points), and well
  ahead of Laya.
- Median time per item: 2.2 s, with 1–3 questions packed into one pass.
- 87 items were longer than 576 tokens and were refused.

**Speed on this laptop**, one request with a 70–550 token state:

| Winnow-12B | Where | Time |
|---|---|---|
| llama.cpp Q8_0 | CPU (12 cores) | ~10–15 s per question |
| w8a16 QNN chain | NPU | 4.2 s per question (weights exceed the NPU mapping limit) |
| LPBQ int4 chain | NPU | 1.6 s per question |
| **LPBQ int4, questions packed into one pass** | **NPU** | **~2.2 s per request** (all questions of a request that fit in 576 tokens) |

## How it works

```
request (state + questions)
  │  winnow_prompt.py   Python port of Winnow's prompt rendering + tokenizer (token-exact with the C++ server)
  ▼
prefix tokens (system text + state)  +  one block per question
  │  pack: prefix once, then every question block; block attention mask (a question sees the prefix and itself);
  │        positions restart at the end of the prefix for each question
  ▼
embeddings on the CPU (token_embd rows × √3840)
  │
  ▼  12 × QNN context (4 decoder layers each, static 576-token graph)
  │     weights: int4 LPBQ (block 32)   activations: uint16   RMSNorm, Gelu, Softmax, MatMul all on the HTP
  ▼
answer head on the CPU: RMSNorm → dot with the answer-letter embedding rows → soft-cap 30 → softmax over the options
```

Why this shape:
- **Static graphs.** The HTP wants fixed shapes.
- **Chunks of 4 layers.** One graph per chunk keeps each compile to about 1.5 minutes and each context under 0.5 GB.
- **One pass for all questions.** Winnow's CPU server evaluates the state once and forks one branch per question. The packed
  block mask gives the same answers in a single NPU pass.

## Step by step

All commands run on Windows in native ARM64 Python 3.12. `WINNOW_HOME` is a working folder; the scripts live in
`winnow-npu/scripts` and look for `..\models`, `..\onnx` and `..\src` relative to `WINNOW_HOME`.

1. **Environments.** Run `winnow-npu\setup_envs.ps1`. It creates:
   - `build`: onnxruntime-qnn 1.24.4, for building and quantizing;
   - `run`: onnxruntime 1.30 + onnxruntime-qnn 2.6.0, for compiling and serving. The run env is **required** for LPBQ int4
     on the NPU; see the traps.
2. **Model.** Download `EldanRing/Winnow-12B` at revision `b6ac22b0` into `models\Winnow-12B`:
   - `gguf/Winnow-12B-Q8_0.gguf` (check it against the repo's `SHA256SUMS`);
   - `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` into `models\Winnow-12B\tok`.
3. **gguf-py.** Clone `EldanRing/winnow-inference` at `6c2b3c0` into `src\winnow-inference`. Then fetch its pinned llama.cpp
   commit (`911f6cdc`) into `.runtime\llama.cpp`; the scripts use only its pure-Python `gguf-py`.
4. **Check the tokenizer port** (optional, needs the CPU Winnow server; see `winnow-npu/cpu`). Run `tools\collect_ref.py` and
   compare the ids with `winnow_prompt.encode`.
5. **Build and calibrate** (build env, about 2–3 h, about 25 GB of temporary disk):
   ```
   python scripts\prepare_chain.py --calib-jsonl CALIB.jsonl --out chain_pub --ncal 32
   ```
   `CALIB.jsonl` is one request per line (`{"state": ..., "questions": {...}}`). Use text that looks like what you will ask
   about. We used the public synthetic suite `harness/items/suite-v1.jsonl`.
6. **4-bit, packing and length** (build env, about 15 min):
   ```
   python scripts\convert_chain_lpbq.py chain_pub
   python scripts\make_packed_chunks.py --chain chain_pub
   python scripts\resize_packed.py --chain chain_pub --seq 576
   ```
7. **Compile for the NPU** (run env, about 20 min, once): `python scripts\compile_chain.py --chain chain_pub --seq 576`
8. **Serve** (run env, in a new process): `python scripts\npu_serve.py --chain chain_pub --seq 576 --port 8014`. Then:
   ```
   curl -s http://127.0.0.1:8014/v1/systemone -H "Content-Type: application/json" -d "{\"state\": \"Outlook keeps asking for my password\", \"questions\": {\"queue\": {\"type\": \"choice\", \"instructions\": \"Which team?\", \"criteria\": {\"accounts\": null, \"email\": null, \"devices\": null}}}}"
   ```

## The traps, and what fixed each one

| Symptom | Cause | Fix |
|---|---|---|
| Every RMSNorm ran on the CPU and each layer split into 5 NPU partitions | The QNN EP (1.24.4) does not place the contrib op `SimplifiedLayerNormalization` | Export RMSNorm as ONNX **`RMSNormalization` (opset 23)**. It runs natively on the HTP and was the most accurate of three formulations tried. |
| About 2% of extra error came from the MLP after 16-bit activation quantization | A written-out tanh-GELU (`x³`, `tanh`, `+1`) has a huge dynamic range | Use the ONNX **`Gelu`** op (`approximate="tanh"`). It is native on the HTP. |
| 28% relative error per layer | Per-channel int4: one scale across 3,840 inputs, and outliers ruin it | Block-wise int4 |
| 72% error on the NPU with block-wise int4 (the same model gives 14.6% on the ORT CPU) | Plain `DequantizeLinear(block_size=…)` weights are not what the HTP implements | **LPBQ**: int4 weights whose block scales come from a second `DequantizeLinear` over **uint8 per-block integers × a float per output channel** (`lpbq.py`). That is the pattern `lpbqmatmul_fusion.cc` fuses. |
| LPBQ compiles but outputs garbage (124% error) | QNN reads int4 LPBQ block scales as **4-bit** (`blockScaleBitwidth = 4`); values above 15 are misread | Keep per-block integers in **1..15** and set the per-channel scale to max/15. |
| `Failed to create tensor … error code: 1000` for every LPBQ weight | onnxruntime-qnn **1.24.4** (QNN 2.3x) rejects LPBQ tensors | Use **onnxruntime 1.30 + onnxruntime-qnn 2.6.0** (plugin EP, QNN 2.39 / HTP backend 2.50) for compiling and running |
| `QNN graph execute error. Error code: 1003` on the first run | Running straight after compiling all 12 chunks in the same process | Compile in one process (`compile_chain.py`) and serve from the cached contexts in a fresh one. Loading takes about 1–2 s per chunk. |
| The w8a16 chain was twice as slow as its per-chunk timings suggest | The NPU keeps only about **9–10 GB** of weights mapped, **system-wide**. 12 × 0.9 GB forces re-mapping on every pass. Two processes do not help. | 4-bit LPBQ (about 5.6 GB) fits. Per-chunk time is unchanged, but the chain runs at the sum of its parts. |
| A request with two questions took 2 × 1.6 s | One pass per question, and the shared state processed twice | Pack all questions into one pass: RoPE tables and mask become graph **inputs**; block mask; positions restart per question (`make_packed_chunks.py`, `npu_winnow.py`) |
| Packed requests did not fit 512 tokens | A state plus two long questions often exceeds 512 tokens | `resize_packed.py` changes the static length (576) by editing Reshape targets. It needs no rebuild or recalibration. |
| The server answered correctly at first, then silently went wrong: benchmark answers at chance level, and each request *faster* than before, with no errors | A threading HTTP server ran every request on a new thread. Calling the QNN sessions (ORT 1.30 + onnxruntime-qnn 2.6.0) from many different threads corrupts results after roughly 60–120 requests. The same requests on one thread ran clean. | **Call the NPU from one thread only.** `npu_serve.py` is single-threaded and runs a start-up self-test. Test with a canary before and after a few hundred varied requests. |
| Mid-run: `NPU crashed. SSR detected during QNN graph execute`, then every request failed | The NPU subsystem restarted (seen here during a long unattended run) and the loaded sessions were dead | `npu_serve.py` catches NPU errors, reloads every session, re-runs the self-test and retries the request once. It exits if the self-test fails. |
| The ONNX model failed to load: `Unsupported model IR version: 14` | onnx 1.23 writes IR 14 by default; ORT 1.24 reads up to 13 | `make_model(..., ir_version=10)` |

CPU side (for the reference server and for checking):
- **Winnow's server needs a GPU:** it refuses to start with "requires a GPU backend". A two-line CPU fallback,
  `cpu/local-cpu-fallback.patch`, fixes that; the device is only used for memory figures.
- **The first request crashed** with access violation 0xc0000005. llama.cpp's CPU weight repacking cannot read the tied
  embedding back, which Winnow's answer head needs. Start with `--no-repack`.
- **This llama.cpp has no `--no-mmap`:** use `--load-mode none` to avoid holding the weights twice.
- **Build llama.cpp with clang,** CPU only: MSVC is rejected for ARM64. On this laptop, the Adreno OpenCL backend coincided
  with two blue screens, so it stays off.

## From 7 s to 2 s

These timings are for the compiled graphs, whatever the request content (static shapes), plus the measured JevBench runs:

| Step | Time | Why it helped |
|---|---|---|
| CPU, llama.cpp Q8_0 | 6–8 s per JevBench easy item | Baseline; grows with prompt length and option count |
| NPU, w8a16 | 4.2 s per 512-token pass | The HTP does the maths; slowed by NPU weight re-mapping (11 GB of weights) |
| NPU, LPBQ int4 | 1.6 s per 512-token pass | Fits the NPU's mapping limit (5.6 GB) |
| NPU, LPBQ int4, every question in one 576-token pass | 2.2 s per request (measured on JevBench: 2.1–2.2 s per item) | The state is processed once; one pass per request instead of one per question |

## Checking your build

- **Phase 1:** `tools/winnow_ref.py` rebuilds Winnow in PyTorch straight from the GGUF, one layer at a time, and compares the
  answer logits with the CPU server. We got the same answer on 6/6 questions, with max |Δp| 0.03 on a near-tie.
- **Tokenizer:** `winnow_prompt.py` gave the same token ids as the C++ server on 229/229 requests.
- **One layer on the NPU:**
  - `tools/phase2_layer.py` runs one layer in fp32 on the CPU, in fp16 on the NPU, and as w8a16 or int4 on the NPU;
  - `tools/lpbq_layer_test.py` runs the LPBQ variants.
  - Expect fp16 at about 8e-4 relative error, w8a16 at about 2%, and LPBQ block 32 at about 12%.
- **The server:** it should give the same answers as the packed chain it was compiled from. Here it gave 20/20, with
  |Δp| = 0.

## decider-12b: the same recipe

[Mapika/decider-12b](https://huggingface.co/Mapika/decider-12b) v2 is Gemma-4-12B-it with a merged LoRA: the same
architecture as Winnow-12B. The pipeline above builds it unchanged ([`decider-npu/`](../decider-npu)).

**What carried over unchanged:**
- **Conversion:** Mapika ships bf16 safetensors only. llama.cpp's converter (`Gemma4UnifiedForConditionalGeneration`)
  made a Q8_0 GGUF with exactly Winnow's tensor layout: 667 tensors, K=V global layers, the proportional-RoPE table,
  soft-cap 30.
- **The NPU build:** chunking, LPBQ 4-bit, packing, the 576-token resize and compiling all ran unchanged.

**What is Decider's own:**
- **Prompt:** decider-ai 1.8.1's chat layout (`decider.serve.prepare`). The answer slot is the last token of each question
  row, so the packed passes work as they do for Winnow.
- **Answer head:** `softcap30(h · embedding[letter])` divided by Decider's temperature for the answer type (choice 1.5,
  yes/no 0.05, score 1.0).
- **Calibration:** the same public suite, rendered in Decider's prompt format (`decider12b_calib_seqs.py`).

**Checks and results** (NPU, 4-bit, about 2.2 s per request):
- **Against a PyTorch reference** rebuilt from the GGUF: the same answer on 6/6 rows, median logit difference 1.15.
- **JevBench public items:** easy 1.000, standard **0.986** (level with Jev).
  - On the 50 hard items that also fit Winnow's longer prompt: **0.700**, against Jev 0.760 and Winnow NPU 0.680.
  - Decider's shorter prompt fits 53 hard items; on those it scores 0.717.
  - Mapika reports 0.712 on the full hard tier at full precision; that covers different items.
- **The private helpdesk-ticket job** in [LAYA-FINETUNE.md](LAYA-FINETUNE.md): 77.5% bucket and 77.5% work type at
  2.3 s per ticket, slightly ahead of Winnow NPU (75.5% and 73.5%).

**NPU files:** [tielmane/decider-12b-NPU-LPBQ-X-Elite](https://huggingface.co/tielmane/decider-12b-NPU-LPBQ-X-Elite).

## Limits

- **Length:** 576 tokens per pass, for the state plus the questions in that pass. Questions that do not fit go into extra
  passes, and a state plus one question over 576 tokens returns HTTP 422. Longer inputs need larger graphs; attention cost
  grows with the square of the length.
- **Accuracy:** 4-bit rounding moves close calls. The NPU answer matched CPU Winnow on 64 of 67 JevBench easy and standard
  items, but only 41 of 50 hard ones: 0.680 against 0.800 on the CPU. Classification survives 4 bits; multi-step reasoning
  loses some. Better 4-bit rounding (GPTQ-style) and more calibration data are the obvious next steps.
- **Memory:** about 6 GB of NPU-mapped weights plus about 1 GB of process memory. Do not run it next to another large NPU
  model.
- **Hardware:** tested on one X Elite (HTP v73) only. Newer chips (X2 Elite, v81) should work but are untested here.
- **Jev is still faster and more accurate:** about 0.2 s per request through TypeSafe's API. This is a local, offline
  alternative, not a replacement.

## Credits and licences

- **Winnow-12B** by EldanRing: Apache-2.0, fine-tuned from Google's Gemma 4 12B.
- **winnow-inference** by EldanRing, on llama.cpp (MIT).
- **ONNX Runtime and its QNN execution provider** by Microsoft (MIT), with Qualcomm's QNN runtime libraries shipped in the
  `onnxruntime-qnn` wheels.
- **JevBench** by Benchmark Heaven (fstandhartinger/jevbench).
- The public benchmark data comes from its original sources, rebuilt by `harness/fairbench`.
- The code in this folder is Apache-2.0, like the rest of this repository.
