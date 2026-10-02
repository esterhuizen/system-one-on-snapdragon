# decider-npu: Mapika's decider-12b on the Snapdragon Hexagon NPU

[decider-12b](https://huggingface.co/Mapika/decider-12b) v2 is Gemma-4-12B-it with a merged LoRA, the same architecture as
Winnow-12B. So the [`winnow-npu`](../winnow-npu) pipeline builds it unchanged. Only the prompt and the answer head are
Decider's own. The steps, traps and numbers are in [../docs/WINNOW-NPU.md](../docs/WINNOW-NPU.md) ("decider-12b"), and the
ready-made NPU files are at [tielmane/decider-12b-NPU-LPBQ-X-Elite](https://huggingface.co/tielmane/decider-12b-NPU-LPBQ-X-Elite).

| File | What it does |
|---|---|
| `npu_decider.py` | Runtime. Builds prompts with decider-ai's `decider.serve.prepare` (chat layout); runs the 12 LPBQ chunks with all of a request's question rows packed behind one shared prefix; answers with `softcap30(h · embedding[letter]) / T_type` and `decider.systemone.assemble`. |
| `npu_decider_serve.py` | Jev-compatible `POST /v1/systemone` on 127.0.0.1:8016. Single-threaded, start-up self-test, reload after an NPU crash. |
| `decider12b_calib_seqs.py` | Calibration token sequences in Decider's prompt format, from a JSONL of requests (we used `harness/items/suite-v1.jsonl`). |
| `decider12b_refseqs.py`, `decider12b_ref.py` | A PyTorch reference rebuilt from the GGUF, one layer at a time: answer logits for six public JevBench rows. |
| `decider12b_npu_check.py` | NPU chain against that reference (we got 6/6 same answer, median logit difference 1.15). |

**Environment variables:**
- `WINNOW_HOME`: the winnow-npu working folder (onnx/, src/ with the pinned llama.cpp gguf-py).
- `DECIDER_MODELS`: the folder holding `decider-12b/` (the HF files) and `decider-12b-gguf/`.
- `JEVBENCH_PUBLIC`: the JevBench public dataset folder (for the reference rows).
- `DECIDER_NPU_RAW=1` (analysis only): each answer also carries `x_raw_logits`, the soft-capped letter logits before the
  temperature, so other temperatures can be scored offline (`harness/noul_temperature.py`).

**Build** (the winnow-npu scripts, with two extra arguments):
```
python convert_hf_to_gguf.py %DECIDER_MODELS%\decider-12b --outtype q8_0 --outfile %DECIDER_MODELS%\decider-12b-gguf\decider-12b-v2-Q8_0.gguf
python decider-npu\decider12b_calib_seqs.py --items harness\items\suite-v1.jsonl --model-dir %DECIDER_MODELS%\decider-12b --out calib_seqs.json
python winnow-npu\scripts\prepare_chain.py --out chain_decider12b --calib-seqs calib_seqs.json --gguf <the GGUF> --ncal 32
python winnow-npu\scripts\convert_chain_lpbq.py chain_decider12b
python winnow-npu\scripts\make_packed_chunks.py --chain chain_decider12b
python winnow-npu\scripts\resize_packed.py --chain chain_decider12b --seq 576
python winnow-npu\scripts\compile_chain.py --chain chain_decider12b --seq 576
python decider-npu\npu_decider_serve.py
```
**Environment** (Python 3.12 arm64): decider-ai 1.8.1, transformers, fastapi and uvicorn, onnxruntime 1.30.0 and
onnxruntime-qnn 2.6.0, onnx, numpy, and torch from the PyTorch CPU index.
