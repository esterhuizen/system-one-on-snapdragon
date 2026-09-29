@echo off
rem Build llama-quantize from the same pinned tree, CPU only (reuses the existing .build configuration).
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64 || exit /b 1
if "%WINNOW_HOME%"=="" set WINNOW_HOME=%~dp0..
cmake --build %WINNOW_HOME%\src\winnow-inference\.build --target llama-quantize -j 6 || exit /b 1
echo BUILD-OK
