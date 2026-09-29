@echo off
rem Build the pinned, patched Winnow server (llama-server + /v1/systemone) for Windows ARM64, CPU ONLY.
rem Every GPU backend is off: llama.cpp's Adreno OpenCL backend coincided with two BSODs on this laptop.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64 || exit /b 1
if "%WINNOW_HOME%"=="" set WINNOW_HOME=%~dp0..
set S=%WINNOW_HOME%\src\winnow-inference
set LLVM=C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/Llvm/ARM64/bin
cmake -S %S% -B %S%\.build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER="%LLVM%/clang.exe" -DCMAKE_CXX_COMPILER="%LLVM%/clang++.exe" ^
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF -DGGML_HIP=OFF -DGGML_SYCL=OFF ^
  -DGGML_BLAS=OFF -DGGML_RPC=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF || exit /b 1
cmake --build %S%\.build --target llama-server winnow-unit -j 6 || exit /b 1
%S%\.build\winnow-unit.exe || exit /b 1
echo BUILD-OK
