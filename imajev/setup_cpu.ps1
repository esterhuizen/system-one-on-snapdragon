$ErrorActionPreference = 'Stop'
$UV = 'uv.exe'; $R = '$env:USERPROFILE\local-imajev'; $PY = "$R\envs\cpu\Scripts\python.exe"
function Run { & $args[0] @($args[1..($args.Count-1)]); if ($LASTEXITCODE -ne 0) { throw "failed: $args" } }
if (-not (Test-Path $PY)) { Run $UV venv "$R\envs\cpu" --python '$env:USERPROFILE\AppData\Local\Programs\Python\Python312-arm64\python.exe' }
Run $UV pip install --python $PY --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu
Run $UV pip install --python $PY transformers==5.17.0 huggingface_hub==1.33.0 tokenizers==0.23.2 safetensors==0.8.0 numpy==2.5.3 Jinja2==3.1.6 fastapi==0.141.1 uvicorn==0.54.0 httpx==0.28.1 peft==0.21.0 accelerate==1.15.0 "pillow>=11" "pydantic>=2,<3" "python-multipart>=0.0.9"
Run $UV pip install --python $PY --no-deps -e "$R\src\imajev"
& $UV pip freeze --python $PY | Out-File -Encoding ascii "$R\envs\cpu.lock.txt"
Run $PY -c "import torch, transformers, peft, vision_decision; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'imajev ok')"
