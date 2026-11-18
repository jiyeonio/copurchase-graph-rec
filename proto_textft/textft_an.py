"""Partial fine-tuning of the text encoder (last 2 KR-ELECTRA layers trained with the recommender, runs <ds>11/T_textft_text and
T_textft_text_co_lgcn_late) against the fixed text embedding of the paper (<ds>11/I_text, I_text_co_lgcn_late) and, on Instacart, ID only
(ic11/K_id). Appendix H, Table A11.
Paired tests per user (seed-averaged NDCG@10, analysis/common.py test), 1:49 and full-catalogue ranking, all users and the target groups of
analysis/link_an_lg.py (prev = purchased before, linked = new and linked by a co-purchase edge, unlinked = new, no edge):
  ft|<model>             delta = fine-tuned - fixed, model text or text_co_lgcn_late
  graphgain|ft           delta = Text+Graph - Text only within fine-tuning
  graphgain|fixed        delta = Text+Graph - Text only with the fixed embedding (paper runs)
  graphgain|ft-fixed     delta = (TG - T)_fine-tuned - (TG - T)_fixed, per user
  vsid|<model>           (Instacart) delta = fine-tuned model - ID only (paper run K_id)
plus the learned fusion weights of Text+Graph and the seed results of every run.
Inputs: <PU> paper dumps (<ds>_text_*, <ds>_text_co_lgcn_late_*, ic_id_fix_*), <PU2> dumps of the fine-tuned runs, <RUNS> result_U_seed*.json.
usage: python proto_textft/textft_an.py [--datasets kc ic] [--seeds 42 ...]   -> <AN>/textft_an[_seeds_<S>].json"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
from common import *
from link_an_lg import feats

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=["kc", "ic"])
ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
a = ap.parse_args()
S = a.seeds
GROUPS = ('all', 'prev', 'linked', 'unlinked')


def pus(DS, name, proto, d):
    xs = [json.load(open(f'{d}{DS}_{name}_{proto}_seed{s}.json')) for s in S]
    return {u: float(np.mean([x[u]['n10'] for x in xs])) for u in xs[0]}


out = {"seeds": S}
for DS in a.datasets:
    F, _ = feats(DS)
    for proto in ('1to49', 'full'):
        M = {('fixed', 'text'): pus(DS, 'text', proto, PU), ('fixed', 'tg'): pus(DS, 'text_co_lgcn_late', proto, PU),
             ('ft', 'text'): pus(DS, 'T_textft_text', proto, PU2), ('ft', 'tg'): pus(DS, 'T_textft_text_co_lgcn_late', proto, PU2)}
        if DS == 'ic':
            M['fixed', 'id'] = pus(DS, 'id_fix', proto, PU)
        us = sorted(set.intersection(*[set(v) for v in M.values()]))
        D = {'ft|text': (lambda u: M['ft', 'text'][u], lambda u: M['fixed', 'text'][u]),
             'ft|text_co_lgcn_late': (lambda u: M['ft', 'tg'][u], lambda u: M['fixed', 'tg'][u]),
             'graphgain|ft': (lambda u: M['ft', 'tg'][u], lambda u: M['ft', 'text'][u]),
             'graphgain|fixed': (lambda u: M['fixed', 'tg'][u], lambda u: M['fixed', 'text'][u]),
             'graphgain|ft-fixed': (lambda u: M['ft', 'tg'][u] - M['ft', 'text'][u], lambda u: M['fixed', 'tg'][u] - M['fixed', 'text'][u])}
        if DS == 'ic':
            D.update({'vsid|text': (lambda u: M['ft', 'text'][u], lambda u: M['fixed', 'id'][u]),
                      'vsid|text_co_lgcn_late': (lambda u: M['ft', 'tg'][u], lambda u: M['fixed', 'id'][u]),
                      'vsid|fixed_text': (lambda u: M['fixed', 'text'][u], lambda u: M['fixed', 'id'][u]),
                      'vsid|fixed_text_co_lgcn_late': (lambda u: M['fixed', 'tg'][u], lambda u: M['fixed', 'id'][u])})
        for g in GROUPS:
            idx = us if g == 'all' else [u for u in us if F[u]['rel'] == g]
            if not idx:
                continue
            for k, (x, y) in D.items():
                r = test(np.array([x(u) - y(u) for u in idx]))
                out[f'{DS}|{proto}|{k}|{g}'] = dict(a=float(np.mean([x(u) for u in idx])), b=float(np.mean([y(u) for u in idx])), **r)
            for (c, m), v in M.items():
                out[f'{DS}|{proto}|mean|{c}|{m}|{g}'] = float(np.mean([v[u] for u in idx]))
        for k in D:
            r = out[f'{DS}|{proto}|{k}|all']
            print(DS, proto, k, f"a {r['a']:.4f} b {r['b']:.4f} delta {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {'*' if r['sig'] else ''}")
    for c, run in (('ft', 'T_textft_text_co_lgcn_late'), ('fixed', 'I_text_co_lgcn_late')):
        W = [json.load(open(f'{RUNS}{DS}11/{run}/result_U_seed{s}.json'))['view_weights'] for s in S]
        out[f'{DS}|weights|{c}'] = dict(per_seed={str(s): w for s, w in zip(S, W)}, w_text_mean=float(np.mean([w['text'] for w in W])),
                                        w_text_sd=float(np.std([w['text'] for w in W], ddof=1)) if len(W) > 1 else None,
                                        w_graph_mean=float(np.mean([w['graph'] for w in W])),
                                        w_graph_sd=float(np.std([w['graph'] for w in W], ddof=1)) if len(W) > 1 else None)
        print(DS, 'weights', c, {k: (round(v, 4) if v is not None else None) for k, v in out[f'{DS}|weights|{c}'].items() if k != 'per_seed'})
    runs = [('ft', 'T_textft_text'), ('ft', 'T_textft_text_co_lgcn_late'), ('fixed', 'I_text'), ('fixed', 'I_text_co_lgcn_late')]
    if DS == 'ic':
        runs.append(('fixed', 'K_id'))
    for c, run in runs:
        R = [json.load(open(f'{RUNS}{DS}11/{run}/result_U_seed{s}.json')) for s in S]
        out[f'{DS}|runs|{run}'] = dict(best_epoch=[r['best_epoch'] for r in R], elapsed_min=[r.get('elapsed_min') for r in R],
                                       test_1to49=[r['test']['overall'] for r in R], test_full_masked=[r['test_full_masked']['overall'] for r in R])
name = 'textft_an' + ('' if S == SEEDS else '_seeds_' + '_'.join(map(str, S))) + '.json'
json.dump(out, open(AN + name, 'w'), indent=1); print('->', AN + name)
