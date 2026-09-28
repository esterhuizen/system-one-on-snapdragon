"""Build FAIR-benchmark part C (Banking77, 12 options) -> results/items/fair/banking77.jsonl

Run with the NPU env (needs the exact Laya prompt builder + tokenizer for the head-fit check):
  cd $WIN_HOME/local-laya && envs/npu/Scripts/python.exe -X utf8 results/items/fair/_work/banking77/build.py
Optional:  --tiers  (after measure_items.py) adds "tier" to every item from banking77.measure.json.

Plan: <repo>/docs/fair-benchmark-plan-2026-09-26.md section 3 C.
Source: mteb/banking77 test.jsonl at revision 0fd18e25b25c072e09e0d92ab615fda904d66300 (3,080 rows),
byte-identical (sha256 fb1b0043...) to fairbench-sources/samples/b77_test.jsonl.
Exclusion: every text in sysone-bench's banking77_12 gate (96 items, datasets/public_cases.json
at commit 1ac3650a65e3783ac9615f26b02bc7c1d5ad3b23) is removed from the pool before sampling.
Head fit: if the 12-option question head would be cut by Laya's 192-token head budget (the
measure tool's HEAD-CUT flag), the item's distractors are redrawn with the next per-item seed
("...|redraw1", "|redraw2", ...) - same gold, same uniform draw, descriptions never edited.
"""
import collections, json, random, hashlib, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[3]                                  # .../local-laya/results
ROOT = RESULTS.parent                                      # .../local-laya
SRC = RESULTS / "fairbench-sources" / "samples" / "b77_test.jsonl"
GATE = HERE / "sysone_public_cases.json"
OUT = HERE.parents[1] / "banking77.jsonl"
MEASURE = HERE.parents[1] / "banking77.measure.json"
SEED = 20260926
PER_INTENT = 5
N_DISTRACT = 11
PART = "banking77"
QID = "intent"
INSTR = "What does the customer want help with?"
SRC_SHA = "fb1b0043ded745b8767687084786e6dd0a5f0ce03243b6131992a1c7ae2c2595"
DATASET = ("mteb/banking77@0fd18e25b25c072e09e0d92ab615fda904d66300 test.jsonl "
           "(PolyAI banking77 test split, 3080 rows)")

if "--tiers" in sys.argv:                                  # step 2: stamp tiers from the measured result
    meas = {r["id"]: r for r in json.loads(MEASURE.read_text(encoding="utf-8"))}
    items = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    tiers = collections.Counter()
    for it in items:
        r = meas[it["id"]]
        assert not r["head_cut"] and not r["no_npu"], it["id"]
        Ls = [x["L"] for x in r["rows"]]
        t = ("trunc" if max(Ls) > 512 else "t128" if max(Ls) <= 128 else
             "t256" if min(Ls) > 128 and max(Ls) <= 256 else
             "t512" if min(Ls) > 256 and max(Ls) <= 512 else "other")
        it["tier"] = t; tiers[t] += 1
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("tiers:", dict(tiers)); sys.exit(0)

sys.path.insert(0, str(ROOT / "src" / "laya-snapdragon"))
from laya_snapdragon.common import build_prefix           # noqa: E402
from laya_snapdragon.tokenizer import Tokenizer           # noqa: E402
MODELS = ROOT / "models" / "npu"
tok = Tokenizer(MODELS / "laya" / "tokenizer")
HEAD = json.loads((MODELS / "laya" / "rl_agent_config.json").read_text())["head_max_len"]


def head_cut(criteria):                                    # identical test to measure_items.py
    q = {"t": "choice", "ins": INSTR, "crit": criteria}
    return len(build_prefix(tok, q, head_max_len=10_000)[0]) > len(build_prefix(tok, q, HEAD)[0])


raw = SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "source file changed"
rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
assert len(rows) == 3080


def norm(t):
    return " ".join(t.lower().split())


gate = json.loads(GATE.read_text(encoding="utf-8"))["suites"]["banking77_12"]["cases"]
assert len(gate) == 96
gate_texts = {norm(c[0]["text"]) for c in gate}

# label id -> raw label name; key = lower-cased label name (only 'Refund_not_showing_up' changes)
labels = {}
for r in rows:
    labels.setdefault(r["label"], r["label_text"])
assert len(labels) == 77
key_of = {lid: name.lower() for lid, name in labels.items()}
assert len(set(key_of.values())) == 77

pool = collections.defaultdict(list)          # label id -> [row index] in file order, gate texts removed
n_excluded = 0
for i, r in enumerate(rows):
    if norm(r["text"]) in gate_texts:
        n_excluded += 1
        continue
    pool[r["label"]].append(i)

rng = random.Random(SEED)
picked = []
for lid in sorted(labels):                     # intents in label-id order 0..76
    picked += rng.sample(pool[lid], PER_INTENT)
rng.shuffle(picked)                            # interleave intents in the output file

all_keys = [key_of[l] for l in sorted(labels)]
items, redrawn = [], []
for n, i in enumerate(picked, 1):
    r = rows[i]
    gold = key_of[r["label"]]
    iid = f"{PART}-{n:04d}"
    for attempt in range(50):
        seed = f"{SEED}|{PART}|{i}" + (f"|redraw{attempt}" if attempt else "")
        irng = random.Random(seed)             # per-item seed (string seed -> sha512, stable across runs)
        distract = irng.sample([k for k in all_keys if k != gold], N_DISTRACT)
        opts = [gold] + distract
        irng.shuffle(opts)
        criteria = {k: k.replace("_", " ") for k in opts}
        if not head_cut(criteria):
            break
    else:
        raise SystemExit(f"{iid}: no head-fitting draw")
    if attempt:
        redrawn.append((iid, i, gold, attempt))
    items.append({
        "id": iid, "suite": PART, "domain": PART, "format": "message",
        "state": r["text"],
        "questions": {QID: {"type": "choice", "instructions": INSTR, "criteria": criteria}},
        "gold": {QID: gold},
        "meta": {QID: {"difficulty": None, "evidence_position": "n/a", "orig_type": "choice",
                       "source_label": r["label_text"], "twin_of": None}},
        "src": {"dataset": DATASET, "row": str(i), "date": None, "slice": "test"},
    })

# sanity checks
assert len(items) == 77 * PER_INTENT
assert len({it["state"] for it in items}) == len(items)
assert not any(norm(it["state"]) in gate_texts for it in items)
c = collections.Counter(it["gold"][QID] for it in items)
assert set(c.values()) == {PER_INTENT} and len(c) == 77
for it in items:
    cr = it["questions"][QID]["criteria"]
    assert len(cr) == 12 and it["gold"][QID] in cr

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")

pos = collections.Counter(list(it["questions"][QID]["criteria"]).index(it["gold"][QID]) for it in items)
print("gate texts:", len(gate_texts), "rows excluded:", n_excluded)
print("pool sizes min/max:", min(map(len, pool.values())), max(map(len, pool.values())))
print("items:", len(items), "gold position histogram:", dict(sorted(pos.items())))
print("redrawn for head fit:", redrawn)
print("wrote", OUT)
