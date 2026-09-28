@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64 || exit /b 1
set R=%USERPROFILE%\local-imajev
set PY=%R%\envs\cpu\Scripts\python.exe
"uv.exe" pip install --python %PY% setuptools wheel ninja || exit /b 1
set FORCE_CUDA=0
set TORCHVISION_USE_PNG=0
set TORCHVISION_USE_JPEG=0
set TORCHVISION_USE_WEBP=0
set TORCHVISION_USE_HEIC=0
set TORCHVISION_USE_AVIF=0
set TORCHVISION_USE_NVJPEG=0
set TORCHVISION_USE_VIDEO_CODEC=0
set TORCHVISION_USE_FFMPEG=0
set MAX_JOBS=6
set DISTUTILS_USE_SDK=1
cd /d %R%\src\torchvision
"uv.exe" pip install --python %PY% --no-build-isolation --no-deps -v . || exit /b 1
%PY% -c "import torch, torchvision; print('torchvision', torchvision.__version__, 'ops ok', torch.ops.torchvision.nms is not None)"
