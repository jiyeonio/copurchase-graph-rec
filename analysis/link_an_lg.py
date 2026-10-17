"""Test users by (i) the relation of the target to the history (purchased before / new and linked by a co-purchase edge / new and not
linked) and (ii) the co-purchase degree of the target: seed-averaged NDCG@10 of every model and paired tests, for the real graph and
the rewired control, under 1:49 ranking, full-catalogue ranking and full-catalogue ranking without removing history items.
-> <AN>/link_an_lg.json"""
import collections, json, numpy as np
from common import *
MODELS = {'id': 'id_fix', 'text': 'text', 'graph': 'co_lgcn', 'final': 'text_co_lgcn_late', 'final_rw': 'text_co_lgcn_late_rwlg', 'graph_rw': 'co_lgcn_rwlg'}
PAIRS = [('text', 'final'), ('graph', 'final'), ('id', 'final'), ('final_rw', 'final'), ('text', 'final_rw'), ('graph_rw', 'graph')]
MIN_USERS = 30
def feats(DS):
    """-> {user: relation of the target, degree of the target}, share of items without an edge"""
    hist, target, trh = data(DS, max_orders=None); nbr = cograph(trh, FMIN[DS]); F = {}
    for u in hist:
        H = {i for o in hist[u] for i in o}; t = target[u]
        F[u] = dict(rel='prev' if t in H else ('linked' if set(nbr[t]) & H else 'unlinked'), deg=len(nbr[t]))
    return F, float(np.mean([len(nbr[i]) == 0 for i in range(N[DS])]))
def paired(d):
    r = test(d); del r['n']; return r
if __name__ == '__main__':
    OUT = {}
    for DS in ('kc', 'ic'):
        F, iso = feats(DS); OUT[f'{DS}|isolated_items'] = iso
        for proto in ('1to49', 'full', 'unmasked'):
            M = {}
            for k, n in MODELS.items():
                if proto == 'unmasked':          # no item removed: NDCG@10 from the rank dumps (not made for the rewired control)
                    if k.endswith('_rw'): continue
                    d = [json.load(open(f"{RK}{DS}_{ {'id_fix': 'id'}.get(n, n) }_unmasked_seed{s}.json")) for s in SEEDS]
                    M[k] = {u: float(np.mean([(1 / np.log2(x[u]['rank'] + 2) if x[u]['rank'] < 10 else 0.0) for x in d])) for u in d[0]}
                else:
                    M[k] = pu(DS, n, proto, PU2 if k.endswith('_rw') else PU)
            users = sorted(set.intersection(*[set(v) for v in M.values()]))
            degs = np.array([F[u]['deg'] for u in users]); q = np.quantile(degs[degs > 0], [1 / 3, 2 / 3])
            G = collections.OrderedDict([('all', lambda f: True), ('prev', lambda f: f['rel'] == 'prev'), ('linked', lambda f: f['rel'] == 'linked'), ('unlinked', lambda f: f['rel'] == 'unlinked'),
                                         ('deg0', lambda f: f['deg'] == 0), ('deglow', lambda f: 0 < f['deg'] <= q[0]), ('degmid', lambda f: q[0] < f['deg'] <= q[1]), ('deghigh', lambda f: f['deg'] > q[1])])
            OUT[f'{DS}|{proto}|deg_cuts'] = [float(x) for x in q]
            for g, fn in G.items():
                idx = [u for u in users if fn(F[u])]; OUT[f'{DS}|{proto}|{g}|n'] = len(idx)
                if len(idx) < MIN_USERS: continue
                for k in M: OUT[f'{DS}|{proto}|{g}|{k}'] = float(np.mean([M[k][u] for u in idx]))
                for a, b in PAIRS:
                    if a in M and b in M: OUT[f'{DS}|{proto}|{g}|{b}-{a}'] = paired(np.array([M[b][u] - M[a][u] for u in idx]))
    json.dump(OUT, open(AN + 'link_an_lg.json', 'w'), indent=0)
    print('written')
