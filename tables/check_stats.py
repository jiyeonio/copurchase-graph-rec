"""Statistics of the two datasets (the dataset statistics table and the statistics stated in the text) from the raw data. -> <PAPER>/tables/check_stats.json"""
import json, collections, itertools, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import *
out = {}
K = KC
def graph(orders, fmin, N):
    cnt = collections.Counter()
    for o in orders:
        for a, b in itertools.combinations(sorted(set(o)), 2): cnt[(a, b)] += 1
    deg = np.zeros(N, int); E = 0
    for (a, b), c in cnt.items():
        if c >= fmin: deg[a] += 1; deg[b] += 1; E += 1
    return E, deg
def ostats(hists):
    npu = [len(h) for h in hists]; ipo = [len(o) for h in hists for o in h]; ipo_d = [len(set(o)) for h in hists for o in h]
    return dict(orders_per_user_mean=float(np.mean(npu)), orders_per_user_median=float(np.median(npu)), items_per_order_mean=float(np.mean(ipo)), items_per_order_median=float(np.median(ipo)),
                items_per_order_distinct_mean=float(np.mean(ipo_d)), single_item_order_share=float(np.mean(np.array(ipo_d) == 1)), multi_item_order_share=float(np.mean(np.array(ipo_d) > 1)))
tr = json.load(open(K + 'train_data.json'))
for nm in ('train', 'val', 'valid', 'test'):
    for suf in ('_history.json', '_data.json'):
        if os.path.exists(K + nm + suf): d = json.load(open(K + nm + suf)); out[f'kc_{nm}{suf[:-5]}_users'] = len(d)
hk = [[[int(i) for i in o['items']] for o in tr[u]] for u in tr]
out['kc_train_data'] = ostats(hk)
if os.path.exists(K + 'train_history.json'):
    th = json.load(open(K + 'train_history.json')); out['kc_train_history'] = ostats([[[int(i) for i in o['items']] for o in th[u]] for u in th])
for f in (1, 2, 3):
    E, deg = graph([o for h in hk for o in h], f, N_ITEMS["kc"]); out[f'kc_graph_f{f}'] = dict(edges=E, mean_deg=float(deg.mean()), median_deg=float(np.median(deg)), mean_deg_nonisolated=float(deg[deg > 0].mean()), median_deg_nonisolated=float(np.median(deg[deg > 0])), isolated=float((deg == 0).mean()))
I = IC
sp = json.load(open(I + 'split.json')); h = json.load(open(I + 'user_item_history.json')); tg = json.load(open(I + 'target.json'))
out['ic_users'] = {k: len(v) for k, v in sp.items()}; out['ic_items_max_id'] = int(max(max(i for o in hh for i in o) for hh in h.values()))
hi = [h[str(u)] for u in sp['train']]; out['ic_train'] = ostats(hi)
for f in (1, 3):
    E, deg = graph([o for hh in hi for o in hh], f, N_ITEMS["ic"]); out[f'ic_graph_f{f}'] = dict(edges=E, mean_deg=float(deg.mean()), median_deg=float(np.median(deg)), mean_deg_nonisolated=float(deg[deg > 0].mean()), median_deg_nonisolated=float(np.median(deg[deg > 0])), isolated=float((deg == 0).mean()))
json.dump(out, open(PAPER + 'tables/check_stats.json', 'w'), indent=1); print(json.dumps(out, indent=1))
