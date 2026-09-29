"""Phase 3 step 2: build, calibrate and quantize the 48-layer chain as 12 chunks of 4 layers (w8a16 QDQ for QNN HTP).

For chunk k: build fp32 ONNX (build_chunk_onnx.py) -> run the calibration sequences through it on ORT CPU (fp32; its
outputs become chunk k+1's calibration inputs) -> quantize with the Phase 2 recipe (uint16 activations, int8 symmetric
per-channel MatMul weights, other initializers uint16) -> delete the fp32 chunk (disk). Resumable per chunk.
Calibration sequences: one per question (prefix + question block, <= S tokens) from --calib-jsonl, shuffled, first --ncal.

python prepare_chain.py --calib-jsonl CALIB.jsonl --out chain_pub [--chunks 0-11] [--ncal 32] [--layers-per-chunk 4]
"""
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np
import onnx
import onnxruntime as ort

W = os.environ.get("WINNOW_HOME") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); S = 512; D = 3840
sys.path.insert(0, os.path.join(W, r"src\winnow-inference\.runtime\llama.cpp\gguf-py"))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--chunks", default="0-11"); ap.add_argument("--ncal", type=int, default=32)
ap.add_argument("--layers-per-chunk", type=int, default=4)
ap.add_argument("--calib-jsonl", required=True, help="Jev-format requests (state + questions) to calibrate on; tokenized with winnow_prompt.py")
ap.add_argument("--out", default="chain", help="output folder under onnx\\ (and calib\\<out>)")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
lo, hi = map(int, a.chunks.split("-")); LPC = a.layers_per_chunk
CAL = os.path.join(W, "calib", a.out); OX = os.path.join(W, "onnx", a.out); os.makedirs(CAL, exist_ok=True); os.makedirs(OX, exist_ok=True)


def calib_inputs0():
    p = os.path.join(CAL, "in_00.npy")
    if os.path.exists(p):
        return
    seqs = []
    import random
    sys.path.insert(0, os.path.join(W, "scripts")); import winnow_prompt as wp
    for r in map(json.loads, open(a.calib_jsonl, encoding="utf-8")):
        e = wp.encode({"state": r["state"], "questions": r["questions"]})
        seqs += [e["prefix_ids"] + s for s in e["suffix_ids"] if len(e["prefix_ids"]) + len(s) <= S]
    random.Random(a.seed).shuffle(seqs)
    seqs = seqs[: a.ncal]
    assert all(len(s) <= S for s in seqs), "a calibration sequence exceeds S"
    t = gguf.GGUFReader(os.path.join(W, r"models\Winnow-12B\gguf\Winnow-12B-Q8_0.gguf"))
    E = next(x for x in t.tensors if x.name == "token_embd.weight"); raw = np.asarray(E.data).reshape(int(E.shape[1]), -1)
    X = np.zeros((len(seqs), S, D), np.float32)
    for i, s in enumerate(seqs):
        X[i, : len(s)] = dequantize(raw[np.asarray(s)], E.tensor_type).reshape(len(s), D) * np.sqrt(D)
    np.save(p, X); np.save(os.path.join(CAL, "lengths.npy"), np.array([len(s) for s in seqs]))
    print(f"[calib] {len(seqs)} sequences, lengths {min(map(len, seqs))}-{max(map(len, seqs))} -> {p}", flush=True)


calib_inputs0()
for k in range(lo, hi + 1):
    qdir = os.path.join(OX, f"c{k:02d}_w8a16"); fdir = os.path.join(OX, f"c{k:02d}_fp32")
    nxt = os.path.join(CAL, f"in_{k + 1:02d}.npy")
    if os.path.exists(os.path.join(qdir, "model.onnx")) and os.path.exists(nxt):
        print(f"[chunk {k}] done already", flush=True); continue
    t0 = time.perf_counter()
    X = np.load(os.path.join(CAL, f"in_{k:02d}.npy"))
    if not os.path.exists(os.path.join(fdir, "model.onnx")):
        subprocess.run([sys.executable, "-X", "utf8", os.path.join(W, "scripts", "build_chunk_onnx.py"), "--start", str(k * LPC),
                        "--end", str((k + 1) * LPC), "--seq", str(S), "--out-dir", fdir], check=True)
    so = ort.SessionOptions(); so.intra_op_num_threads = 12
    s = ort.InferenceSession(os.path.join(fdir, "model.onnx"), so, providers=["CPUExecutionProvider"])
    Y = np.stack([s.run(["y"], {"x": X[i: i + 1]})[0][0] for i in range(len(X))]); del s
    t1 = time.perf_counter()
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

    class R(CalibrationDataReader):
        def __init__(self): self.it = iter([{"x": X[i: i + 1]} for i in range(len(X))])
        def get_next(self): return next(self.it, None)
    m = onnx.load(os.path.join(fdir, "model.onnx"), load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    over = {nd.input[1]: [{"quant_type": QuantType.QInt8, "symmetric": True, "axis": 1}]
            for nd in m.graph.node if nd.op_type == "MatMul" and nd.input[1] in inits}
    cfg = get_qnn_qdq_config(os.path.join(fdir, "model.onnx"), R(), activation_type=QuantType.QUInt16,
                             weight_type=QuantType.QUInt16, per_channel=True, init_overrides=over)
    cfg.use_external_data_format = True
    os.makedirs(qdir, exist_ok=True)
    quantize(os.path.join(fdir, "model.onnx"), os.path.join(qdir, "model.onnx"), cfg)
    np.save(nxt, Y)
    shutil.rmtree(fdir)
    print(f"[chunk {k}] layers {k * LPC}-{(k + 1) * LPC - 1}: fp32 propagate {t1 - t0:.0f}s, quantize {time.perf_counter() - t1:.0f}s; "
          f"output max|y| {np.abs(Y).max():.1f}", flush=True)
