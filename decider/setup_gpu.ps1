$ErrorActionPreference = 'Stop'
$UV = 'uv.exe'; $R = '$env:USERPROFILE\local-decider'; $PY = "$R\envs\gpu-qnn\Scripts\python.exe"
function Run { & $args[0] @($args[1..($args.Count-1)]); if ($LASTEXITCODE -ne 0) { throw "failed: $args" } }
if (-not (Test-Path $PY)) { Run $UV venv "$R\envs\gpu-qnn" --python '$env:USERPROFILE\AppData\Local\Programs\Python\Python312-arm64\python.exe' }
Run $UV pip install --python $PY --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu
Run $UV pip install --python $PY transformers==5.17.0 huggingface_hub==1.33.0 tokenizers==0.23.2 safetensors==0.8.0 numpy==2.5.3 Jinja2==3.1.6 onnx==1.23.0 onnxscript onnxruntime==1.30.0 onnxruntime-qnn==2.6.0 ml_dtypes
Run $UV pip install --python $PY --no-deps --require-hashes -r "$R\envs\decider-ai.req.txt"
& $UV pip freeze --python $PY | Out-File -Encoding ascii "$R\envs\gpu-qnn.lock.txt"
Run $PY -c "import onnxruntime as o, onnxruntime_qnn as q, decider; o.register_execution_provider_library('QNNExecutionProvider', q.get_library_path()); print(o.__version__, [(d.ep_name, str(d.device.type)) for d in o.get_ep_devices()])"
