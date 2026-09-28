import os
import json, collections
F=os.path.expandvars("$WIN_HOME/local-laya/results/items/fair/cfpb.jsonl")
M=os.path.expandvars("$WIN_HOME/local-laya/results/items/fair/cfpb.measure.json")
m={r['id']: r for r in json.load(open(M, encoding='utf-8'))}
def tier(r):
    Ls = [x['L'] for x in r['rows']]
    if max(Ls) > 512: return 'trunc'
    if max(Ls) <= 128: return 't128'
    if min(Ls) >= 257: return 't512'
    return 't256'   # all rows 129..256, or (5 A2 items) noul row 129 + twin row 128 -> needs the 256 bucket
out = []
for line in open(F, encoding='utf-8'):
    it = json.loads(line)
    t = tier(m[it['id']])
    new = {}
    for k, v in it.items():
        if k != 'tier': new[k] = v
        if k == 'format': new['tier'] = t
    out.append(new)
with open(F, 'w', encoding='utf-8', newline='\n') as f:
    for it in out: f.write(json.dumps(it, ensure_ascii=False) + '\n')
print(collections.Counter(it['tier'] for it in out))
