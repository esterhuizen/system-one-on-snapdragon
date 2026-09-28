# setup_env.ps1 -Name <cpu|gpu-qnn|harness|npu|npu26|gpu-dml>  : create one native ARM64 Python 3.12 venv per the install plan
param([Parameter(Mandatory=$true)][string]$Name)
$ErrorActionPreference = 'Stop'
$R  = '$env:USERPROFILE\local-laya'; $PY = '$env:USERPROFILE\AppData\Local\Programs\Python\Python312-arm64\python.exe'
$EnvDir = "$R\envs\$Name"
if (Test-Path "$EnvDir\Scripts\python.exe") { Write-Output "env $Name already exists"; } else { & $PY -m venv $EnvDir; if ($LASTEXITCODE -ne 0) { throw "venv $Name failed ($LASTEXITCODE)" } }
$E = "$EnvDir\Scripts\python.exe"
function Pip { & $E -m pip @args; if ($LASTEXITCODE -ne 0) { throw "pip $args failed ($LASTEXITCODE)" } }
& $E -c "import sysconfig; p=sysconfig.get_platform(); assert p=='win-arm64', p; print(p)"
if ($LASTEXITCODE -ne 0) { throw "$E is not a native win-arm64 interpreter" }
Pip install --disable-pip-version-check --upgrade pip
switch ($Name) {
  'cpu'     { Pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
              Pip install "laya[serve]==0.3.20" }
  'gpu-qnn' { Pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
              Pip install "laya[serve]==0.3.20" onnx==1.23.0 onnxscript onnxruntime==1.30.0 onnxruntime-qnn==2.6.0 }
  'gpu-dml' { Pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
              Pip install "laya[serve]==0.3.20" onnx==1.23.0 onnxruntime-windowsml==1.30.0.202609102321 }
  'harness' { Pip install httpx==0.28.1 "typesafe-sdk==0.7.1" numpy }
  'npu'     { Pip install -e "$R\src\laya-snapdragon"
              Pip install --no-deps "laya==0.3.20"
              Pip install "fastapi>=0.110" "uvicorn>=0.27" "python-multipart>=0.0.9" }
  'npu26'   { Pip install onnxruntime==1.30.0 onnxruntime-qnn==2.6.0 onnx==1.23.0 "numpy>=1.26" "tokenizers>=0.20" "huggingface_hub>=0.25" ml_dtypes
              Pip install --no-deps -e "$R\src\laya-snapdragon"
              Pip install --no-deps "laya==0.3.20"
              Pip install "fastapi>=0.110" "uvicorn>=0.27" "python-multipart>=0.0.9" }
  default   { throw "unknown env $Name" }
}
& $E -m pip freeze | Out-File -Encoding utf8 "$R\envs\$Name.lock.txt"
Write-Output "ENV $Name OK"
