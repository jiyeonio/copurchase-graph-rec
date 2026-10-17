"""Two stratified comparisons:
 C  Katcher's: proposed model minus the item-level variant by the mean number of items per order of the user's history
 E  gain of Text+Graph over Graph within popularity thirds of the target, lower / upper half by degree; rank correlation of degree and popularity
-> <EXTRA>/an2.json"""
import json, numpy as np
from common import *
MIN_USERS = 30; OUT = {}
for DS in ('kc', 'ic'):
    hist, target, trh = data(DS); nbr = cograph(trh, FMIN[DS])
    pop = np.zeros(N[DS])
    for hh in trh:
        for o in hh:
            for i in o: pop[i] += 1
    msz = {u: float(np.mean([len(o) for o in hist[u]])) for u in hist}
    for proto in ('1to49', 'full'):
        F = pu(DS, 'text_co_lgcn_late', proto); V = pu(DS, 'T_itempos_causal', proto, PU2); users = sorted(set(F) & set(V))
        if DS == 'kc':
            for lab, (lo, hi) in zip(['1.0', '1.0-1.5', '>1.5'], [(1, 1.0001), (1.0001, 1.5), (1.5, 1e9)]):
                idx = [u for u in users if lo <= msz[u] < hi]
                if len(idx) >= MIN_USERS: OUT[f'{DS}|C|{proto}|meansize|{lab}|itempos_causal'] = dict(test(np.array([F[u] - V[u] for u in idx])), full=float(np.mean([F[u] for u in idx])))
        G = pu(DS, 'co_lgcn', proto); deg = {u: len(nbr.get(target[u], {})) for u in users}; pp = {u: pop[target[u]] for u in users}
        OUT[f'{DS}|E|{proto}|spearman_deg_pop'] = float(np.corrcoef(np.argsort(np.argsort([deg[u] for u in users])), np.argsort(np.argsort([pp[u] for u in users])))[0, 1])
        q = np.quantile([pp[u] for u in users], [1 / 3, 2 / 3])
        for pi, (lo, hi) in enumerate([(-1, q[0]), (q[0], q[1]), (q[1], 1e12)]):
            idx = [u for u in users if lo < pp[u] <= hi]; md = np.median([deg[u] for u in idx])
            for half, sel in (('lowdeg', [u for u in idx if deg[u] <= md]), ('highdeg', [u for u in idx if deg[u] > md])):
                if len(sel) >= MIN_USERS: OUT[f'{DS}|E|{proto}|pop{pi}|{half}'] = dict(n=len(sel), gain=float(np.mean([F[u] - G[u] for u in sel])), mean_deg=float(np.mean([deg[u] for u in sel])), mean_pop=float(np.mean([pp[u] for u in sel])))
json.dump(OUT, open(EXTRA + 'an2.json', 'w'), indent=0)
for k, v in OUT.items(): print(k, v)
