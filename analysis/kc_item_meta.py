"""Katcher's item names and categories by item index (used by emb_vis2.py). -> <AN>/kc_item_meta.json"""
import csv, json
from common import *
K = KC_SPLIT; OUT = AN + 'kc_item_meta.json'
rows = []
for f in ('train_data.csv', 'val_data.csv', 'test_data.csv'):
    with open(K + f, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['user_id'] and r['order_id'] and r['product_id'] and r['initial_paid_at']: rows.append(r)
pids = sorted({int(float(r['product_id'])) for r in rows}); item2idx = {p: i for i, p in enumerate(pids)}
meta = {}
for r in rows:
    i = item2idx[int(float(r['product_id']))]
    if i not in meta: meta[i] = {'product_id': int(float(r['product_id'])), 'name': r['product_name'], 'category': r['category_name'], 'root_category': r['root_category_name']}
json.dump({str(k): meta[k] for k in sorted(meta)}, open(OUT, 'w'), ensure_ascii=False, indent=0)
print('items', len(pids))
