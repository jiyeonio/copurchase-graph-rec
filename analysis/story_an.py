"""Katcher's: gain of Text+Graph over the text alone and over the rewired control
 A  by the relation of the target to the history
 B  among new targets linked to the history: by link strength (thirds) and by whether the linked item is in the last order
 C  by the mean number of items per order of the history
-> <EXTRA>/story_new.json"""
import json, numpy as np
from common import *
DS = 'kc'; OUT = {}
hist, target, trh = data(DS); nbr = cograph(trh, FMIN[DS]); Hs = {u: {i for o in hist[u] for i in o} for u in hist}
st = {}; grp = {}
for u in target:
    t = target[u]
    if t in Hs[u]: grp[u] = 'prev'; continue
    cs = [nbr[t][i] for i in Hs[u] if i in nbr.get(t, {})]
    grp[u] = 'linked' if cs else 'unlinked'
    if cs: st[u] = (sum(cs), any(i in nbr.get(t, {}) for i in hist[u][-1]))       # link strength, link to the last order
msz = {u: float(np.mean([len(o) for o in hist[u]])) for u in hist}
for proto in ('1to49', 'full'):
    F = pu(DS, 'text_co_lgcn_late', proto); T = pu(DS, 'text', proto); RW = pu(DS, 'text_co_lgcn_late_rwlg', proto, PU2); users = sorted(set(F) & set(T) & set(RW))
    def cmp(sel):
        m = lambda D: float(np.mean([D[u] for u in sel])); d = lambda A, B: test([A[u] - B[u] for u in sel])
        return dict(n=len(sel), text=m(T), real=m(F), rw=m(RW), real_text=d(F, T), real_rw=d(F, RW), rw_text=d(RW, T))
    for g in ('all', 'prev', 'linked', 'unlinked'):
        OUT[f'A|{proto}|{g}'] = cmp([u for u in users if g == 'all' or grp[u] == g])
    us = [u for u in users if u in st]; key = np.array([st[u][0] for u in us]); q = np.quantile(key, [1 / 3, 2 / 3])
    for lab, sel in (('weak', [u for u, k in zip(us, key) if k <= q[0]]), ('medium', [u for u, k in zip(us, key) if q[0] < k <= q[1]]), ('strong', [u for u, k in zip(us, key) if k > q[1]]),
                     ('last', [u for u in us if st[u][1]]), ('earlier', [u for u in us if not st[u][1]])):
        OUT[f'B|{proto}|{lab}'] = dict(cmp(sel), cuts=[float(x) for x in q])
    for lab, (lo, hi) in zip(['1.0', '1.0-1.5', '>1.5'], [(1, 1.0001), (1.0001, 1.5), (1.5, 1e9)]):
        OUT[f'C|{proto}|meansize|{lab}'] = cmp([u for u in users if lo <= msz[u] < hi])
json.dump(OUT, open(EXTRA + 'story_new.json', 'w'), indent=0); print('wrote', len(OUT))
