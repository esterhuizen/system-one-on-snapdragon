# Contributions to upstream projects

Anything that belongs to a model's own code goes to its repository, where it stays maintained. The first three were posted on 2026-09-29, and the Winnow-work reports on 2026-10-01 (links below). The llama.cpp
blue-screen report is still a draft.

## piffie/laya-snapdragon: hardware report (X Elite, HTP v73)

Posted: https://github.com/piffie/laya-snapdragon/issues/1

**Title:** Works on Snapdragon X Elite X1E80100 (HTP v73), with latency and parity numbers

**Configuration:**
- Windows 11 26200 ARM64, native CPython 3.12, laya-snapdragon v0.2.0, onnxruntime-qnn 1.24.4.
- `build --seq 128 256 512` compiled fp16 contexts in about 1 minute each, about 90 s for 512.
- The compiler reports about 4 GB of VTCM spill at 512 tokens.

**Verification:** `verify` gave 12/12 identical answers vs the CPU model, maximum |dp| 4.1e-3.

**Per-question latency:**

| Tokens | NPU | ORT CPU |
|---|---|---|
| 128 | 33 ms | 143–175 ms |
| 256 | 100 ms | 424 ms |
| 512 | 385 ms | 751–1083 ms |

**Also worked:**
- 24-option contexts (s128/256/512 × m24). The 128-token one costs about 37 ms.
- The same encoder, exported fp32 static, runs on the Adreno through onnxruntime-qnn 2.6.0's GPU backend (see below).

## Mapika/decider: Windows ARM64 notes (issue plus docs PR)

Posted: https://github.com/Mapika/decider/issues/18

**Title:** Windows on ARM (Snapdragon X): numpy<2 pin blocks install; bfloat16 on CPU is 13× slower; llama.cpp needs clang

1. **numpy pin.** `numpy<2`: numpy 1.26 has no win_arm64 cp312 wheel, so pip tries to build it from source. The only runtime numpy
   use (`engine.py`, `np.full`) works on numpy 2. Could the pin be relaxed?
2. **CPU defaults to bfloat16.** On Windows ARM64 PyTorch 2.14 that runs about 13× slower than float32 (decider-2b: 137 s vs 11 s
   for a 3-question request). Suggest defaulting to float32 on CPU when there's no fast bf16 path, or documenting it.
3. **GGUF path.** llama-cpp-python 0.3.35 only builds with clang on Windows ARM64 (MSVC is rejected by ggml-cpu).
   - The working CMake arguments were CPU-only: `-DGGML_OPENMP=OFF` with the Visual Studio clang, and every GPU backend OFF.
   - Result: decider-2b v11 Q8_0 at about 220 ms per 3-question request on 8 threads (Q4_K_M about 330 ms).
4. **QNN (Snapdragon GPU) export.** A static ONNX export of decider-2b (S=256) unrolls the Gated DeltaNet layers into about
   20,000 nodes. onnxruntime-qnn 2.6.0's GPU backend compiles any piece up to about 3,000 nodes but caps the total per process,
   so the model cannot run there as one graph or as a chain of segments. Details and scripts are in this repo. A compact
   DeltaNet export (a recurrent form rather than an unrolled chunk scan) would be needed for Qualcomm accelerators.
5. **Server binding.** `scripts/serve.sh` binds `0.0.0.0` with no authentication. Consider `127.0.0.1` as the default.

## mohit67890/imajev: torchvision required for text-only use

Posted: https://github.com/mohit67890/imajev/issues/1

**Title:** Text-only CPU use still needs torchvision (Qwen3VLVideoProcessor); no Windows ARM64 wheel

- `AutoProcessor.from_pretrained` loads `Qwen3VLVideoProcessor`, which imports torchvision even for text-only requests.
- torchvision 0.29 has no win_arm64 wheel. We built it from source with MSVC ARM64, CPU-only with codecs off, and imajev-2b then
  ran on CPU in float32 at about 14.5 s per 3-question request.
- Possible fix: construct the processor without the video processor when no images or video are used.

## ggml-org/llama.cpp: possible Adreno OpenCL kernel crash (post only after confirming the driver)

**Title:** BSOD (0xD1) on Snapdragon X Elite when using the Adreno OpenCL backend (b10453), Adreno driver 31.0.133.1

- **Status:** *Do not post until a kernel debugger has identified the faulting module in `C:\Windows\Minidump\092826-*.dmp`
  (admin rights needed).*
- **Circumstances:**
  - Two 0xD1 bugchecks while running the b10453 Windows ARM64 Adreno build, and a build that loaded every backend, with a Qwen3.5
    GGUF (decider-2b / 4b).
  - CPU-only builds are stable.

## onnxruntime/onnxruntime-qnn: LPBQ requirements; cross-thread corruption

Posted to the QNN execution provider's own repository:
- https://github.com/onnxruntime/onnxruntime-qnn/issues/893: int4 LPBQ block scales must be 1..15 (plus the 1.24.4 rejection, `SimplifiedLayerNormalization` placement and plain block-wise int4).
- https://github.com/onnxruntime/onnxruntime-qnn/issues/892: sessions called from many threads silently return wrong results; a single thread fixes it.

Original draft:

1. **LPBQ int4 block scales are read as 4-bit.** `qnn_quant_params_wrapper.cc` sets `blockScaleBitwidth = is_int4 ? 4 : 0`.
   - The `lpbqmatmul_fusion` pattern takes per-block scales as a uint8 initializer dequantized per channel, so 8-bit values
     look valid. Values above 15 are silently misread and the MatMul produces garbage: 124% relative error on a Gemma 4
     layer, against 13.7% once the scales were kept to 1..15.
   - Suggest documenting the range, or rejecting out-of-range scales at fusion time.
2. **onnxruntime-qnn 1.24.4 rejects LPBQ weights.** Every fused weight fails with `Failed to create tensor … error code: 1000`
   on an X Elite (HTP v73). The same model compiles and runs correctly with onnxruntime 1.30 + onnxruntime-qnn 2.6.0.
3. **`SimplifiedLayerNormalization` is not placed on the HTP by 1.24.4, while opset 23 `RMSNormalization` is.** With the
   former, every RMSNorm fell back to the CPU and split a decoder layer into 5 NPU partitions.
4. **Plain block-wise int4 `DequantizeLinear(block_size=…)` → MatMul compiles but is wrong on the HTP.** It gives 72%
   relative error on the NPU against 14.6% on the ORT CPU for the same model. It should either be rejected or emulated.

Reproduction scripts: `winnow-npu/tools/lpbq_layer_test.py`, `lpbq_plugin_test.py`, `rmsnorm_probe.py`.

## EldanRing/winnow-inference: CPU-only use

Posted: https://github.com/EldanRing/winnow-inference/issues/3

- `native/engine.h` throws "Winnow requires a GPU backend" when no GPU or iGPU device exists. The device is only used for its
  description and memory figures, so falling back to the CPU device works (`winnow-npu/cpu/local-cpu-fallback.patch`).
- On the CPU, the selected answer head's `ggml_backend_tensor_get` on the tied embedding hits llama.cpp's repack buffer, which
  has no `get_tensor` (null call, access violation). `--no-repack` fixes it; the launcher could set it on CPU.
- The Q8_0 model ran correctly on Windows ARM64 (clang, CPU only).

## ggml-org/llama.cpp: reading back a tensor from the CPU repack buffer crashes

Not posted: it is already reported as https://github.com/ggml-org/llama.cpp/issues/29701 (closed 2026-09-30).

`ggml_backend_tensor_get` on a tensor that lives in the CPU "repack" extra buffer calls a null `get_tensor`, which is an access
violation. It was hit through winnow-inference's classifier-head patch reading the tied `token_embd` rows (Gemma 4 12B Q8_0,
Windows ARM64, build 11036). It could return an error or de-repack instead.
