"""Katcher's, removal of co-purchase edges: gain of Text+Graph over the text alone when 100 / 75 / 50 / 25% of the edges are kept, by the
relation of the target to the history (groups defined on the full graph). -> <EXTRA>/boost23.json"""
import json, numpy as np
from common import *
DS = 'kc'; OUT = {}
hist, target, trh = data(DS); nbr = cograph(trh, FMIN[DS]); Hs = {u: {i for o in hist[u] for i in o} for u in hist}
grp = {u: 'prev' if target[u] in Hs[u] else ('linked' if any(i in nbr.get(target[u], {}) for i in Hs[u]) else 'unlinked') for u in target}
for proto in ('1to49', 'full'):
    T = pu(DS, 'text', proto); G = {100: pu(DS, 'text_co_lgcn_late', proto)}
    for d in (25, 50, 75): G[100 - d] = pu(DS, f'text_co_drop{d}_late', proto, PU_EDGE)
    users = sorted(T)
    for g in ('all', 'prev', 'linked', 'unlinked'):
        sel = [u for u in users if g == 'all' or grp[u] == g]; m = lambda D: float(np.mean([D[u] for u in sel])); d = lambda A, B: test([A[u] - B[u] for u in sel])
        o = dict(n=len(sel), text=m(T))
        for k in (100, 75, 50, 25): o[f'tg{k}'] = m(G[k]); o[f'gain{k}'] = d(G[k], T)
        for k in (75, 50, 25): o[f'd100_{k}'] = d(G[100], G[k])
        OUT[f'E2|{proto}|{g}'] = o
json.dump(OUT, open(EXTRA + 'boost23.json', 'w'), indent=0)
for k, o in OUT.items(): print(k, {a: (f"{v['delta']:+.4f}{'*' if v['sig'] else ''}" if isinstance(v, dict) else v) for a, v in o.items()})
