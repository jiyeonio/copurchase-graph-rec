"""Paired tests per user of the variants against the proposed model: item-level encoder, training on the last order only, order-state
readout (mean instead of last token) and final-target weight lambda = 1, 4. delta = proposed model - variant. -> <AN>/ablation_an.json"""
import json, numpy as np
from common import *
out = {}
for ds in ('kc', 'ic'):
    for proto in ('1to49', 'full'):
        base = pu(ds, 'text_co_lgcn_late', proto)
        for n in ('itempos_causal', 'nodense'):
            a = pu(ds, 'T_' + n, proto, PU2); us = sorted(set(a) & set(base)); r = test(np.array([base[u] - a[u] for u in us])); del r['n']
            out[f'{ds}|{proto}|{n}'] = dict(variant=float(np.mean([a[u] for u in us])), full=float(np.mean([base[u] for u in us])), **r)
            print(ds, proto, n, f"variant {out[f'{ds}|{proto}|{n}']['variant']:.4f}  proposed - variant {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {'*' if r['sig'] else ''}")
# order-state readout (run T_readout_mean: mean hidden state of the order's tokens instead of the last token). Not part of the Holm set.
# Also by the mean number of items per order of the test user's history
# (= 1 / at least 1.5, as for the item-level variant in analysis/extra_an2.py; groups with fewer than 30 users are left out).
for ds in ('kc', 'ic'):
    hist, _, _ = data(ds); msz = {u: float(np.mean([len(o) for o in hist[u]])) for u in hist}
    for proto in ('1to49', 'full'):
        base = pu(ds, 'text_co_lgcn_late', proto); a = pu(ds, 'T_readout_mean', proto, PU2); us = sorted(set(a) & set(base))
        r = test(np.array([base[u] - a[u] for u in us])); n_ = r.pop('n')
        out[f'{ds}|{proto}|readout_mean'] = dict(variant=float(np.mean([a[u] for u in us])), full=float(np.mean([base[u] for u in us])), n=n_, **r)
        print(ds, proto, 'readout_mean', f"variant {out[f'{ds}|{proto}|readout_mean']['variant']:.4f}  proposed - variant {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {'*' if r['sig'] else ''}")
        for lab, (lo, hi) in zip(['1.0', '>1.5'], [(1, 1.0001), (1.5, 1e9)]):
            idx = [u for u in us if lo <= msz[u] < hi]
            if len(idx) >= 30:
                out[f'{ds}|{proto}|readout_mean|meansize|{lab}'] = dict(test(np.array([base[u] - a[u] for u in idx])), variant=float(np.mean([a[u] for u in idx])), full=float(np.mean([base[u] for u in idx])))
            else:
                out[f'{ds}|{proto}|readout_mean|meansize|{lab}'] = dict(n=len(idx), omitted='fewer than 30 users')
# weight lambda of the final-target term (runs T_finalw1, T_finalw4; the proposed model has lambda = 2). Not part of
# the Holm set. All users and the target groups of analysis/link_an_lg.py (purchased before / new and linked / new and not linked).
from link_an_lg import feats
for ds in ('kc', 'ic'):
    F, _ = feats(ds)
    for proto in ('1to49', 'full'):
        base = pu(ds, 'text_co_lgcn_late', proto)
        for lam in (1, 4):
            a = pu(ds, f'T_finalw{lam}', proto, PU2); us = sorted(set(a) & set(base))
            for g in ('all', 'prev', 'linked', 'unlinked'):
                idx = us if g == 'all' else [u for u in us if F[u]['rel'] == g]
                r = test(np.array([base[u] - a[u] for u in idx]))
                out[f'{ds}|{proto}|finalw{lam}' + ('' if g == 'all' else f'|group|{g}')] = dict(variant=float(np.mean([a[u] for u in idx])), full=float(np.mean([base[u] for u in idx])), **r)
            r = out[f'{ds}|{proto}|finalw{lam}']
            print(ds, proto, f'finalw{lam}', f"variant {r['variant']:.4f}  proposed - variant {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {'*' if r['sig'] else ''}")
json.dump(out, open(AN + 'ablation_an.json', 'w'), indent=1)
