$ErrorActionPreference = 'Stop'
$UV = 'uv.exe'; $R = '$env:USERPROFILE\local-decider'; $PY = "$R\envs\cpu\Scripts\python.exe"
function Run { & $args[0] @($args[1..($args.Count-1)]); if ($LASTEXITCODE -ne 0) { throw "failed: $args" } }
if (-not (Test-Path $PY)) { Run $UV venv "$R\envs\cpu" --python '$env:USERPROFILE\AppData\Local\Programs\Python\Python312-arm64\python.exe' }
Run $PY -c "import sysconfig,sys; p=sysconfig.get_platform(); assert p=='win-arm64',p; print(p, sys.version.split()[0])"
Run $UV pip install --python $PY --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu
Run $UV pip install --python $PY transformers==5.17.0 huggingface_hub==1.33.0 tokenizers==0.23.2 safetensors==0.8.0 numpy==2.5.3 Jinja2==3.1.6 fastapi==0.141.1 uvicorn==0.54.0 httpx==0.28.1
Set-Content -Encoding ascii "$R\envs\decider-ai.req.txt" 'decider-ai==1.5.0 --hash=sha256:f72af210ba442a40f7e80339419ea9aa811bec2d1bf4eafe63b392c3cb22d7cb'
Run $UV pip install --python $PY --no-deps --require-hashes -r "$R\envs\decider-ai.req.txt"
& $UV pip freeze --python $PY | Out-File -Encoding ascii "$R\envs\cpu.lock.txt"
Run $PY -c "import decider, torch, transformers; print('decider', getattr(decider,'__version__','?'), 'torch', torch.__version__, 'transformers', transformers.__version__)"
