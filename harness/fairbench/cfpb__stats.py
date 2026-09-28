import json, collections, sys
sys.path.insert(0,'.')
for sl in ('S1','S2'):
    c=collections.Counter(); iss=collections.Counter(); months=collections.Counter()
    for line in open(f'cand_{sl}.jsonl',encoding='utf-8'):
        r=json.loads(line); n=r['ntok']
        b='T128' if 60<=n<=128 else 'T512' if 250<=n<=450 else None
        months[r['date'][:4] if sl=='S1' else r['date'][:7]]+=1
        if not b: continue
        c[(r['product'],b)]+=1
        if r['product']=='Debt collection': iss[(r['issue'],b)]+=1
    print(sl, sorted(months.items()))
    for k,v in sorted(c.items()): print('  ',k,v)
    print('  DC issues:')
    for k,v in sorted(iss.items()): print('    ',k,v)
