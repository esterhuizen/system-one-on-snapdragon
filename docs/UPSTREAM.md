# Contributions to upstream projects

Anything that belongs to a model's own code goes to its repository, where it stays maintained. These are drafts; nothing here has
been posted yet.

## piffie/laya-snapdragon: hardware report (X Elite, HTP v73)

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
