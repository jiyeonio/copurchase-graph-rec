"""Shared loaders and the paired test of the analyses."""
import json, collections, itertools, os, sys, numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import *
N = N_ITEMS
def data(DS, max_orders=50):
    """-> hist {u: [orders]} and target {u: item} of the test users, train_hist (list of the histories of the training users).
    max_orders: Instacart histories are cut to the last max_orders orders (None = all orders)"""
    if DS == 'kc':
        th = json.load(open(KC + 'test_history.json')); td = json.load(open(KC + 'test_data.json')); tr = json.load(open(KC + 'train_data.json'))
        hist = {u: [sorted(set(int(i) for i in o['items'])) for o in th[u]] for u in th}; target = {u: int(td[u]['items'][0]) for u in td}
        trh = [[sorted(set(int(i) for i in o['items'])) for o in tr[u]] for u in tr]
    else:
        sp = json.load(open(IC + 'split.json')); tg = json.load(open(IC + 'target.json')); h = json.load(open(IC + 'user_item_history.json'))
        hist = {str(u): [sorted(set(o)) for o in h[str(u)]][-(max_orders or 0):] for u in sp['test']}; target = {str(u): int(tg[str(u)]) for u in sp['test']}
        trh = [[sorted(set(o)) for o in h[str(u)]] for u in sp['train']]
    return hist, target, trh
def cograph(trh, fmin):
    """co-purchase graph of the training users: nbr[a][b] = number of orders that contain both a and b (pairs with at least fmin orders)"""
    cnt = collections.Counter()
    for hh in trh:
        for o in hh:
            for a, b in itertools.combinations(o, 2): cnt[(a, b)] += 1
    nbr = collections.defaultdict(dict)
    for (a, b), c in cnt.items():
        if c >= fmin: nbr[a][b] = c; nbr[b][a] = c
    return nbr
def test(d):
    """paired differences per user -> mean, bootstrap 95% interval, two-sided Wilcoxon p (zeros excluded), significance"""
    d = np.asarray(d, float); rng = np.random.default_rng(BOOT_SEED); n = len(d); b = d[rng.integers(0, n, (BOOT_N, n))].mean(1); lo, hi = np.percentile(b, [2.5, 97.5])
    nz = d[np.abs(d) > 1e-15]; p = float(wilcoxon(nz).pvalue) if len(nz) > 10 else float('nan')
    return dict(n=n, delta=float(d.mean()), lo=float(lo), hi=float(hi), p=p, sig=bool((lo > 0 or hi < 0) and p < 0.05))
def pu(DS, name, proto, d=PU):
    """seed-averaged per-user NDCG@10 of one run (proto: 1to49 | full)"""
    xs = [json.load(open(f'{d}{DS}_{name}_{proto}_seed{s}.json')) for s in SEEDS]; return {u: float(np.mean([x[u]['n10'] for x in xs])) for u in xs[0]}
