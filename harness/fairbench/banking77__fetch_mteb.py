import json, urllib.request, time
rows=[]
off=0
while True:
    url=f"https://datasets-server.huggingface.co/rows?dataset=mteb/banking77&config=default&split=test&offset={off}&length=100"
    for attempt in range(5):
        try:
            d=json.load(urllib.request.urlopen(url, timeout=60)); break
        except Exception as e:
            print('retry',off,e); time.sleep(3)
    rs=d['rows']
    if not rs: break
    for r in rs: rows.append({"row_idx":r['row_idx'],**r['row']})
    off+=len(rs)
    if off>=d['num_rows_total']: break
json.dump(rows,open('mteb_b77_test.json','w'))
print(len(rows), rows[0], d.get('features'))
