# Fine-tuning Laya into a dedicated classifier with Claude Opus labels

This guide turns [Laya](https://huggingface.co/convaiinnovations/laya), a small open "System One" decision model, into a
dedicated classifier for one fixed job: sorting helpdesk tickets into 12 buckets and 6 work types. The result:
- **Accuracy:** it beat TypeSafe's Jev on that job, 88.5% against 83.5% for the bucket.
- **Speed and cost:** it runs on a Snapdragon X Elite's NPU at about **17 ms per ticket**, offline and free per ticket.
- **Effort:** about 10 minutes of automated Claude Opus labelling and one overnight CPU training run.

**In one sentence:** Claude Opus labelled about 4,800 past tickets, and Laya's encoder, with a new classification head, was
fine-tuned on those labels. That is distillation from a large "teacher" model into a small, fast "student".

The data is a private set of 4,963 real helpdesk tickets. Only aggregate results are published here; no ticket text or
names. The code in [`laya-finetune/`](../laya-finetune) is generic: it works for any fixed set of categorical questions
about short texts.

## Results (200 held-out tickets, never used for training)

| Approach | Bucket (12) | Work type (6) | Per ticket | Where it runs |
|---|---|---|---|---|
| Existing helpdesk "Category" field mapped to buckets | 63.0% | – | – | – |
| Laya out of the box (zero-shot, long bucket descriptions) | 63.5% | 67.0% | 0.14 s | NPU |
| Winnow-12B, 4-bit, both questions in one pass ([WINNOW-NPU.md](WINNOW-NPU.md)) | 75.5% | 73.5% | 2.1 s | NPU |
| Winnow-12B, Q8_0 | 78.5% | 81.0% | 29 s | CPU |
| decider-12b v2, 4-bit ([WINNOW-NPU.md](WINNOW-NPU.md#decider-12b-the-same-recipe)) | 77.5% | 77.5% | 2.3 s | NPU |
| Quick test: Laya encoder frozen, linear classifier on Opus labels | 79.5% | 83.0% | – | CPU |
| Jev 1.13.0 (TypeSafe API) | 83.5% | 85.5% | 0.2 s | cloud |
| **Laya fine-tuned on Opus labels** | **88.5%** | **89.0%** | **0.017 s** | **NPU** |
| *(the Opus labels themselves)* | *95.5%* | *94.0%* | – | – |

- **Confidence you can use:** where the fine-tuned model's top probability is at least 0.7 (80% of tickets), it is right
  **96%** of the time. Send the other 20% to a person.
- **The NPU gives the same answers as the CPU** on all 200 held-out tickets (probabilities within 0.008). The whole set of
  4,963 tickets takes about 90 seconds.
- **Caveat: the reference labels are also Claude-made.** The 200 held-out tickets were labelled blind by two Claude labellers
  and an adjudicator, and the training labels came from Claude Opus. Part of the win over Jev may be a shared way of reading
  tickets. A human spot-check of a random sample is the independent confirmation.

## Why out-of-the-box Laya was weak at this

- **It answers a new question every time.** Laya is a general typed-decision model: it gets a question, reads the options'
  descriptions and picks one.
- **The descriptions got cut.** Twelve long bucket descriptions do not fit its 192-token question budget, so it read
  trimmed versions.
- **It had never seen this taxonomy,** or the house conventions for filing tickets: which bucket a mailbox-access request
  goes into, or what a forwarded vendor email counts as.

A specialist trained on a few thousand examples learns those conventions. A zero-shot reader cannot.

## The recipe

### 1. Labels from Claude Opus: two passes plus adjudication (`label_workflow.js`)

1. **Items:** put them in JSONL, one per line, with an id and only the fields the model should read.
2. **Batches:** `python make_batches.py --items items.jsonl --out-dir WORK --holdout 200` makes batches of 100, holds 200
   items out for evaluation, and writes `train_ids.txt` and `holdout_ids.txt`.
3. **Taxonomy and rules:** write `WORK/taxonomy.json` (`{"tasks": {"bucket": {key: definition, ...}, ...}}`) and
   `WORK/GUIDE.md` (labelling rules). [`examples/`](../laya-finetune/examples) has the helpdesk ones used here.
4. **Run the workflow:** in Claude Code, ask to "use a workflow" with `label_workflow.js` and
   `{"dir": "<WORK>", "batches": [...], "tasks": ["bucket", "work_type"]}`.
   - **Pass A** labels each batch in file order.
   - **Pass B** labels it **independently**, in reverse order and with the tasks in reverse order.
   - **An adjudicator** re-decides only the items where A and B disagree.
   - Agents write labels to files and return only small summaries, so the item text never flows back through the main session.

**What it cost here:** 50 batches × 3 agents = 150 agents, about 6.6 million sub-agent tokens and **9.5 minutes**. The two
passes agreed on 95.8% of buckets and 95.1% of work types; the rest were adjudicated.

**Label your held-out set too, then check it.** Here the Opus labels matched the held-out reference on 95.5% (bucket) and
94.0% (work type). Where Opus said "high confidence", it matched 100%.

### 2. Quick test (5 minutes): is it worth fine-tuning?

`finetune.py` prints an `epoch 0` line before any training. That is the frozen encoder with a logistic-regression head fitted
on your labels: the "quick test". Here it already reached 79.5% / 83.0%. If yours is close to the zero-shot score, more
training will not help much, and the labels or the input fields need work first.

### 3. Fine-tune (`finetune.py`, CPU)

```
python finetune.py --items items.jsonl --labels-dir WORK/final --taxonomy WORK/taxonomy.json \
   --train-ids WORK/train_ids.txt --holdout-ids WORK/holdout_ids.txt [--holdout-gold human_labels.jsonl] \
   --text-template "{summary}\nType: {request_type}\nCategory: {helpdesk_category}" --out-dir OUT --epochs 3
```

- **Model:** Laya's ModernBERT-large encoder (394M parameters, as trained by Convai Innovations for decisions), mean
  pooling, and **one linear head per task**. Laya's general decision head is not used.
- **Initialization:** the heads start from the quick-test solution.
- **Training settings:**
  - 3 epochs, batch 16, AdamW;
  - learning rate 2e-5 for the encoder and 5e-4 for the heads, 6% warm-up, then linear decay;
  - label smoothing 0.05;
  - low-confidence Opus labels count less (high 1.0, medium 0.8, low 0.5).
- **Epoch selection:** 10% of the training items form a dev split that picks the best epoch. The held-out set is only
  scored, never used to choose.
- **Time:** about 1.4 hours per epoch for 4,286 short texts (under 64 tokens) on a 12-core Snapdragon X Elite. Here epoch 2
  was best (dev and held-out both peaked there).

### 4. Put it on the NPU (`export_npu.py`, `run_npu.py`)

- **Export:** `export_npu.py` writes a static [1, 64] ONNX graph: encoder, pooling and one softmax output per task. It
  applies the two graph fixes used for Laya's own encoder (`laya/win/fix_allowzero.py`, `laya/win/qnn_fix.py`) and checks
  ONNX Runtime against PyTorch.
- **Run:** `run_npu.py` compiles it once for the Hexagon NPU (fp16, onnxruntime 1.30 + onnxruntime-qnn 2.6.0), then writes
  predictions with per-item timing. It can also score against labels and compare with the CPU.
  - About 1,280 graph nodes land on the NPU and one tiny GatherND on the CPU.
  - **16–17 ms per ticket for both questions.**

Call the NPU from one thread only; see the traps in [WINNOW-NPU.md](WINNOW-NPU.md).

## Terms of use: whose outputs you may train on

- **TypeSafe (Jev):** its Master Customer Agreement §2.3(b) forbids using Jev's output "to perform model distillation,
  train a model to imitate the output of the Services". We did **not** train on Jev's answers.
- **Anthropic (Claude):**
  - **Commercial Terms** §D.4 bar using the services "to build a competing product or service, including to train competing
    AI models". A private ticket classifier does not compete with Claude, and customers own their outputs (§B).
  - **Consumer Terms:** the same restriction is scoped to products that compete with Anthropic's services.
  - Check the terms for your own account and use. Company data belongs on the commercial / API route.

## Limits

- **One fixed job:** it answers only the tasks it was trained on. For new questions use the original Laya (or Winnow).
- **Changing the taxonomy** means relabelling (minutes) and retraining (hours).
- **Only as good as the labels.** Spot-check them, and keep the held-out set separate from anything used to choose a model.
- **Measured on one private dataset,** so treat the gains as an example, not a guarantee.
