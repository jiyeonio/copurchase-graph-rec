"""Paired tests per user of the two variants (item-level encoder, training on the last order only) against the proposed model.
delta = proposed model - variant. -> <AN>/ablation_an.json"""
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
json.dump(out, open(AN + 'ablation_an.json', 'w'), indent=1)
