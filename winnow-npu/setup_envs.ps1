# Two native ARM64 Python 3.12 environments, exactly as tested (Windows 11 ARM64, Snapdragon X Elite X1E80100).
#   build : chunk build, calibration, LPBQ conversion (onnxruntime-qnn 1.24.4 bundles ONNX Runtime + its quantization tools)
#   run   : NPU compile + runtime/server (onnxruntime 1.30 + onnxruntime-qnn 2.6.0 plugin; needed for LPBQ int4 on the NPU)
# Usage: $env:WINNOW_HOME = "C:\path\to\winnow-work"; .\setup_envs.ps1   (needs uv on PATH and python 3.12 arm64 installed)
$ErrorActionPreference = "Stop"
if (-not $env:WINNOW_HOME) { throw "set WINNOW_HOME first" }
$envs = Join-Path $env:WINNOW_HOME "envs"; New-Item -ItemType Directory -Force $envs | Out-Null
uv venv --python 3.12 (Join-Path $envs "build")
uv pip install --python (Join-Path $envs "build\Scripts\python.exe") onnxruntime-qnn==1.24.4 onnx==1.23.0 numpy==2.5.3 tokenizers==0.23.2
uv venv --python 3.12 (Join-Path $envs "run")
uv pip install --python (Join-Path $envs "run\Scripts\python.exe") onnxruntime==1.30.0 onnxruntime-qnn==2.6.0 onnx==1.23.0 numpy==2.5.3 tokenizers==0.23.2 httpx
Write-Host "gguf-py comes from the pinned llama.cpp tree (src\winnow-inference\.runtime\llama.cpp\gguf-py); the scripts add it to sys.path."
