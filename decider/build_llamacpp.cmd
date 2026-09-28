@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64 || exit /b 1
set R=%USERPROFILE%\local-decider
set PY=%R%\envs\cpu\Scripts\python.exe
set UV=uv.exe
set CMAKE_GENERATOR=Ninja
set CMAKE_ARGS=-DCMAKE_C_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/Llvm/ARM64/bin/clang.exe" -DCMAKE_CXX_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/Llvm/ARM64/bin/clang++.exe" -DGGML_OPENMP=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF -DGGML_METAL=OFF -DGGML_HIP=OFF -DGGML_SYCL=OFF -DGGML_KOMPUTE=OFF -DGGML_BLAS=OFF -DGGML_RPC=OFF -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
set CMAKE_BUILD_PARALLEL_LEVEL=6
echo llama-cpp-python==0.3.35 --hash=sha256:1139dbb54509074b70893fab8554e3b079aa9f4d312058ce4018ef0019e3de12> %R%\envs\llama-cpp.req.txt
%UV% pip install --python %PY% --no-deps --no-binary llama-cpp-python --require-hashes -v -r %R%\envs\llama-cpp.req.txt || exit /b 1
%UV% pip install --python %PY% "diskcache>=5.6.1" "typing-extensions>=4.5.0" || exit /b 1
echo BUILD-OK
