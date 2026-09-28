"""Stream CFPB CCDB narrative exports -> per-slice candidate JSONL with Laya token counts.

S1: date received 2015-03-01 .. 2022-11-30 (exports 1-4)
S2: date received 2026-03-01 .. 2026-08-31 (exports 16-21; narratives published through 2026-08-14)
Keeps rows with a non-empty narrative whose Laya state token count is in 40..480 (bands are cut later).
"""
import os
import csv, io, json, sys, zipfile, collections
sys.path.insert(0, os.path.expandvars("$WIN_HOME/local-laya/results/fairbench-sources/pylib"))
from tokenizers import Tokenizer

W = os.path.expandvars("$WIN_HOME/local-laya/results/items/fair/_work/cfpb")
TOK = Tokenizer.from_file(os.path.expandvars("$WIN_HOME/local-laya/models/npu/laya/tokenizer/tokenizer.json"))
TOK.no_truncation(); TOK.no_padding()
MASK = '[MASK]'
csv.field_size_limit(10**9)

SLICES = {
    'S1': (('2015-03-01', '2022-11-30'), ['1_December_2011_through_April_2018', '2_May_2018_through_April_2021',
                                          '3_May_2021_through_October_2022', '4_November_2022_through_August_2023']),
    'S2': (('2026-03-01', '2026-08-31'), ['16_March_2026', '17_April_2026', '18_May_2026', '19_June_2026',
                                          '20_July_2026', '21_August_2026']),
}

def flush(buf, out, stats):
    texts = [r['narrative'].replace(MASK, ' ') for r in buf]
    encs = TOK.encode_batch(texts, add_special_tokens=False)
    for r, e in zip(buf, encs):
        n = len(e.ids); r['ntok'] = n
        stats['tok_' + ('lt40' if n < 40 else 'gt480' if n > 480 else 'in')] += 1
        if 40 <= n <= 480:
            out.write(json.dumps(r, ensure_ascii=False) + '\n')
    buf.clear()

for sl, ((lo, hi), exports) in SLICES.items():
    stats = collections.Counter(); seen = set()
    with open(f'{W}/cand_{sl}.jsonl', 'w', encoding='utf-8') as out:
        buf = []
        for ex in exports:
            fn = f'CCDB_Export_{ex}'
            z = zipfile.ZipFile(f'{W}/zips/{fn}.zip')
            with z.open(f'{fn}.csv') as fh:
                rd = csv.DictReader(io.TextIOWrapper(fh, encoding='utf-8', newline=''))
                for row in rd:
                    stats['rows'] += 1
                    d = row['Date received'][:10]
                    if not (lo <= d <= hi):
                        continue
                    stats['in_window'] += 1
                    nar = row['Consumer complaint narrative'] or ''
                    if not nar.strip():
                        continue
                    stats['narr'] += 1
                    cid = row['Complaint ID']
                    if cid in seen:
                        stats['dup_id'] += 1; continue
                    seen.add(cid)
                    if len(nar) < 100 or len(nar) > 4000:   # loose char prefilter (bands are 60..450 tokens)
                        stats['char_out'] += 1; continue
                    buf.append({'cid': cid, 'date': d, 'product': row['Product'], 'sub_product': row['Sub-product'],
                                'issue': row['Issue'], 'sub_issue': row['Sub-issue'], 'company': row['Company'],
                                'export': fn, 'narrative': nar})
                    if len(buf) >= 20000:
                        flush(buf, out, stats)
            print(sl, ex, dict(stats), file=sys.stderr, flush=True)
        if buf:
            flush(buf, out, stats)
    print(sl, 'FINAL', dict(stats), file=sys.stderr, flush=True)
    json.dump(dict(stats), open(f'{W}/extract_stats_{sl}.json', 'w'))
