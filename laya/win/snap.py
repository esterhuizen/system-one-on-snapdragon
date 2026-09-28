"""python -X utf8 snap.py [--adsp] <laya_snapdragon CLI args, e.g. --models DIR build --seq 128 256>"""
import importlib.util, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[:1] == ["--adsp"]:
    args = args[1:]
    s = importlib.util.find_spec("onnxruntime_qnn")
    d = Path(s.origin).parent if s else Path(importlib.util.find_spec("onnxruntime").origin).parent / "capi"
    os.environ["ADSP_LIBRARY_PATH"] = str(d) + (";" + os.environ["ADSP_LIBRARY_PATH"] if os.environ.get("ADSP_LIBRARY_PATH") else "")
    print("ADSP_LIBRARY_PATH =", os.environ["ADSP_LIBRARY_PATH"], flush=True)
from laya_snapdragon.__main__ import main
sys.exit(main(args))
