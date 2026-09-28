"""Select CFPB narratives for A1 (product choice) and A2 (debt not owed yes/no + A/B twin) and write items.

Plan: <repo>/docs/fair-benchmark-plan-2026-09-26.md section 3.A
"""
import os
import ast, hashlib, json, random, re, sys, zlib, collections
import numpy as np
sys.path.insert(0, os.path.expandvars("$WIN_HOME/local-laya/results/fairbench-sources/pylib"))
from tokenizers import Tokenizer
TOK = Tokenizer.from_file(os.path.expandvars("$WIN_HOME/local-laya/models/npu/laya/tokenizer/tokenizer.json"))
TOK.no_truncation(); TOK.no_padding()
BYTES_RE = re.compile(r"""^\s*b(['"]).*\1\s*$""", re.S)
DECODED = collections.Counter()

W = os.path.expandvars("$WIN_HOME/local-laya/results/items/fair/_work/cfpb")
OUT = os.path.expandvars("$WIN_HOME/local-laya/results/items/fair/cfpb.jsonl")
PART = 'cfpb'
BANDS = {'T128': (60, 128), 'T512': (250, 450)}
N_A1 = 20          # per class per band per slice
N_A2 = 40          # per label per band per slice
SUB = 3000         # per-stratum random subsample that is MinHash-deduplicated before sampling
SIM = 0.8          # MinHash similarity threshold
NPERM = 128

# ---- A1 gold mapping (CFPB Product -> key); anything else (incl. Debt or credit management / Other financial service) dropped
PRODUCT_MAP = {
    'Credit reporting': 'credit_reporting',
    'Credit reporting, credit repair services, or other personal consumer reports': 'credit_reporting',
    'Credit reporting or other personal consumer reports': 'credit_reporting',
    'Debt collection': 'debt_collection',
    'Credit card': 'credit_card', 'Credit card or prepaid card': 'credit_card', 'Prepaid card': 'credit_card',
    'Checking or savings account': 'bank_account', 'Bank account or service': 'bank_account',
    'Money transfer, virtual currency, or money service': 'money_transfer', 'Money transfers': 'money_transfer',
    'Virtual currency': 'money_transfer',
    'Mortgage': 'mortgage',
    'Vehicle loan or lease': 'consumer_loan', 'Consumer Loan': 'consumer_loan', 'Student loan': 'consumer_loan',
    'Payday loan': 'consumer_loan', 'Payday loan, title loan, or personal loan': 'consumer_loan',
    'Payday loan, title loan, personal loan, or advance loan': 'consumer_loan',
}
A1_Q = 'Which financial product is this complaint about?'
A1_CRIT = {
    'credit_reporting': 'Credit reports, credit scores, or a credit bureau',
    'debt_collection': 'A debt collector or the collection of a debt',
    'credit_card': 'A credit card or prepaid card',
    'bank_account': 'A checking or savings account',
    'money_transfer': 'Sending or receiving money, payment apps, or virtual currency',
    'mortgage': 'A home mortgage or home equity loan',
    'consumer_loan': 'A vehicle, student, payday, or personal loan',
}
A2_Q = 'Does the consumer say they do not owe the debt that is being collected?'
A2_YES = 'the debt is not theirs, was already paid, came from identity theft, or was discharged'
A2_NO = 'the complaint is about something else, such as collection tactics, notices, or threats'
A2_QID = 'not_owed'
# Issue == "Attempts to collect debt not owed" (2017+ name); pre-Apr-2017 name of the same issue is mapped too.
NOT_OWED_ISSUES = {'Attempts to collect debt not owed', "Cont'd attempts collect debt not owed"}

TEMPLATE_RE = re.compile(r'1681|formally dispute|I am writing to', re.I)

# ---- English filter: function-word ratio
EN = set('''the and to of i a my in that is was for on it with have this not they be me are at as but had from
an or by were been has would will their them an there what when which who you your all if so no do did can
could should because about after before into out up than then also any other our we he she his her its'''.split())
ES = set('''el la los las del que por con para una mi es se al lo su pero como mas más muy cuenta este esta
fue sin sobre entre cuando porque yo le les nos ha han sus tengo pago deuda banco'''.split())
WORD = re.compile(r"[a-záéíóúñü']+")

def english(t):
    """Reject only text that is not identifiably English: Spanish function words outnumber English ones,
    >20% of letters are non-ASCII, or fewer than 3 English function words (pure field lists / redaction runs)."""
    ws = [w for w in WORD.findall(t.lower()) if not re.fullmatch(r'x+', w)]
    en = sum(w in EN for w in ws); es = sum(w in ES for w in ws)
    letters = [c for c in t if c.isalpha()]
    nonascii = sum(ord(c) > 127 for c in letters) / max(1, len(letters))
    return en >= 3 and es <= en and nonascii <= 0.2

# ---- MinHash (word 3-gram shingles, 128 multiply-shift hashes)
rng = np.random.default_rng(20260926)
A = (rng.integers(1, 2**63 - 1, NPERM, dtype=np.uint64) * np.uint64(2) + np.uint64(1))
B = rng.integers(0, 2**63 - 1, NPERM, dtype=np.uint64)
TOKRE = re.compile(r'[a-z0-9]+')

def minhash(t):
    ws = TOKRE.findall(t.lower())
    sh = {' '.join(ws[i:i + 3]) for i in range(max(1, len(ws) - 2))}
    x = np.array([zlib.crc32(s.encode()) for s in sh], dtype=np.uint64)
    with np.errstate(over='ignore'):
        h = (x[:, None] * A[None, :] + B[None, :]) >> np.uint64(32)
    return h.min(axis=0).astype(np.uint32)

def sims(sig, mat):
    return (mat == sig[None, :]).mean(axis=1) if len(mat) else np.zeros(0)

# ---- load candidates
def load(sl):
    out = []
    for line in open(f'{W}/cand_{sl}.jsonl', encoding='utf-8'):
        r = json.loads(line)
        n = r['ntok']
        t = r['narrative']
        if BYTES_RE.match(t):   # S2 export stores non-ASCII narratives as a Python bytes literal: decode it
            try:
                v = ast.literal_eval(t.strip())
                if isinstance(v, bytes):
                    t = v.decode('utf-8', errors='replace')
                    r['narrative'] = t; r['decoded_bytes_repr'] = True
                    n = r['ntok'] = len(TOK.encode(t.replace('[MASK]', ' '), add_special_tokens=False).ids)
                    DECODED[sl] += 1
            except (ValueError, SyntaxError):
                pass
        band = next((b for b, (lo, hi) in BANDS.items() if lo <= n <= hi), None)
        if band is None:
            continue
        r['band'] = band
        out.append(r)
    return out

selected, sel_sigs = [], []           # global (cross-stratum) near-duplicate guard
used = set()
log = collections.defaultdict(dict)

def pick(pool, n, seed, tag):
    """Shuffle pool; MinHash-dedupe a random subsample (cluster rep = earliest received);
    walk reps in shuffled order; skip anything near-duplicate to an already-selected item."""
    pool = [r for r in pool if r['cid'] not in used]
    pool.sort(key=lambda r: int(r['cid']))
    random.Random(seed).shuffle(pool)
    sub = pool[:SUB]
    sigs = np.stack([minhash(r['narrative']) for r in sub]) if sub else np.zeros((0, NPERM), np.uint32)
    parent = list(range(len(sub)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(sub)):
        s = sims(sigs[i], sigs[i + 1:])
        for j in np.nonzero(s >= SIM)[0]:
            a, b = find(i), find(i + 1 + int(j))
            if a != b:
                parent[b] = a
    clusters = collections.defaultdict(list)
    for i in range(len(sub)):
        clusters[find(i)].append(i)
    rep = {min(m, key=lambda k: (sub[k]['date'], int(sub[k]['cid']))) for m in clusters.values()}
    got, skipped_global = [], 0
    for i in range(len(sub)):
        if len(got) >= n:
            break
        if i not in rep:
            continue
        gm = np.stack(sel_sigs) if sel_sigs else np.zeros((0, NPERM), np.uint32)
        if len(gm) and sims(sigs[i], gm).max() >= SIM:
            skipped_global += 1; continue
        r = sub[i]; r['_sig'] = sigs[i]
        got.append(r); used.add(r['cid']); sel_sigs.append(sigs[i])
    log[tag] = {'pool_english': len(pool), 'subsample': len(sub), 'clusters': len(clusters),
                'near_dup_removed_in_subsample': len(sub) - len(clusters), 'skipped_cross_stratum': skipped_global,
                'selected': len(got)}
    print(tag, log[tag], file=sys.stderr, flush=True)
    return got

def twin_yes_key(item_id, qid):
    return 'A' if int(hashlib.sha256((item_id + '|' + qid).encode()).hexdigest(), 16) % 2 == 0 else 'B'

items = []
filt = collections.defaultdict(collections.Counter)
for sl in ('S1', 'S2'):
    cands = load(sl)
    for r in cands:
        r['cls'] = PRODUCT_MAP.get(r['product'])
    eng = []
    for r in cands:
        filt[sl]['in_band'] += 1
        if not english(r['narrative']):
            filt[sl]['non_english'] += 1; continue
        eng.append(r)
    filt[sl]['english'] = len(eng)
    filt[sl]['unmapped_product_dropped'] = sum(1 for r in eng if r['cls'] is None)
    filt[sl]['unmapped_products'] = dict(collections.Counter(r['product'] for r in eng if r['cls'] is None))
    for band in BANDS:
        # A1
        for cls in A1_CRIT:
            pool = [r for r in eng if r['band'] == band and r['cls'] == cls]
            for r in pick(pool, N_A1, f'{PART}|A1|{sl}|{band}|{cls}', f'A1/{sl}/{band}/{cls}'):
                items.append(('A1', sl, band, r))
        # A2 (debt collection only)
        dc = [r for r in eng if r['band'] == band and r['product'] == 'Debt collection']
        for lab in (True, False):
            pool = [r for r in dc if (r['issue'] in NOT_OWED_ISSUES) == lab]
            for r in pick(pool, N_A2, f'{PART}|A2|{sl}|{band}|{lab}', f'A2/{sl}/{band}/{"yes" if lab else "no"}'):
                items.append(('A2', sl, band, r))
    del cands, eng

# stable order: A1 then A2, slice, band, then complaint id
order = {'A1': 0, 'A2': 1}
items.sort(key=lambda x: (order[x[0]], x[1], x[2], int(x[3]['cid'])))
with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
    for n, (task, sl, band, r) in enumerate(items, 1):
        iid = f'{PART}-{n:04d}'
        src = {'dataset': f"CFPB Consumer Complaint Database narratives archive, {r['export']}.zip "
                          "(https://files.consumerfinance.gov/f/documents/)",
               'row': r['cid'], 'date': r['date'], 'slice': sl,
               'template_letter': bool(TEMPLATE_RE.search(r['narrative']))}
        if task == 'A1':
            qs = {'product': {'type': 'choice', 'instructions': A1_Q, 'criteria': dict(A1_CRIT)}}
            gold = {'product': r['cls']}
            meta = {'product': {'difficulty': None, 'evidence_position': 'n/a', 'orig_type': 'choice',
                                'source_label': r['product'], 'twin_of': None,
                                'source_sub_label': r['sub_product'], 'task': 'A1', 'state_band': band}}
        else:
            yes = r['issue'] in NOT_OWED_ISSUES
            yk = twin_yes_key(iid, A2_QID); nk = 'B' if yk == 'A' else 'A'
            ab = {yk: A2_YES, nk: A2_NO}
            qs = {A2_QID: {'type': 'noul', 'instructions': A2_Q, 'criteria': {'true': A2_YES, 'false': A2_NO}},
                  A2_QID + '_ab': {'type': 'choice', 'instructions': A2_Q, 'criteria': {'A': ab['A'], 'B': ab['B']}}}
            gold = {A2_QID: yes, A2_QID + '_ab': yk if yes else nk}
            m = {'difficulty': None, 'evidence_position': 'n/a', 'orig_type': 'noul', 'source_label': r['issue'],
                 'source_sub_label': r['sub_issue'], 'task': 'A2', 'state_band': band}
            meta = {A2_QID: dict(m, twin_of=None), A2_QID + '_ab': dict(m, twin_of=A2_QID)}
        if r.get('decoded_bytes_repr'):
            for m_ in meta.values():
                m_['decoded_bytes_repr'] = True
        item = {'id': iid, 'suite': PART, 'domain': PART, 'format': 'complaint', 'state': r['narrative'],
                'questions': qs, 'gold': gold, 'meta': meta, 'src': src}
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
json.dump({'filters': filt, 'decoded_bytes_repr_in_band_pool': DECODED, 'strata': log}, open(f'{W}/build_log.json', 'w'), indent=1, default=str)
print('wrote', len(items), file=sys.stderr)
