"""Katcher's, validation negatives drawn outside the user's history (model/model_katchers.py --val-exclude-history; runs
<RUNS>kc11/<run>_valnh of scripts/14_valneg_check.sh) against the runs of the paper (<RUNS>kc11/<run>, per-user dumps in <PU>). Test users and
test candidates are those of the paper; only the 49 validation negatives, and so the checkpoint chosen on validation, differ. Seeds 42-46.
 1_side_by_side    mean and sd over seeds: validation 1:49, test 1:49, test full catalogue (history removed); overall / novel / repeat
 2_new_minus_paper same model, new - paper per user (seed-averaged NDCG@10): bootstrap 95% CI over users, Wilcoxon (as tables/make_tables.py)
 3_tg_vs_single    Text+Graph - Text only / Graph only / ID only within the paper runs and within the new runs; all, novel, repeat
 4_three_groups    the comparisons of 3 in the three target groups of Table 8 (analysis/link_an_lg.py: purchased before, new and linked by a
                   co-purchase edge, new and not linked), test of analysis/common.py; the paper rows are checked against <AN>link_an_lg.json
 5_best_epoch      epoch chosen on validation per seed
-> <AN>/valneg_check_an.json"""
import json, os
import numpy as np
from scipy.stats import wilcoxon
from common import RUNS, PU, PU2, AN, SEEDS, BOOT_N, BOOT_SEED, test
from link_an_lg import feats

MODELS = [('I_text_co_lgcn_late', 'text_co_lgcn_late', 'Text+Graph'), ('I_text', 'text', 'Text only'), ('I_co_lgcn', 'co_lgcn', 'Graph only'), ('K_id', 'id_fix', 'ID only')]
RUN = {'paper': RUNS + 'kc11/{run}/result_U_seed{s}.json', 'new': RUNS + 'kc11/{run}_valnh/result_U_seed{s}.json'}
DUMP = {'paper': PU + 'kc_{pu}_{p}_seed{s}.json', 'new': PU2 + 'kc_{pu}_valnh_{p}_seed{s}.json'}
PROTO = ('1to49', 'full')


def res(src, run): return [json.load(open(RUN[src].format(run=run, s=s))) for s in SEEDS]
def dumps(src, pu, p): return [json.load(open(DUMP[src].format(pu=pu, p=p, s=s))) for s in SEEDS]


def paired(A, B, grp=None):
    """B - A on seed-averaged per-user NDCG@10 (as paired() of tables/make_tables.py); grp: users of that target group only"""
    users = sorted(set.intersection(*[set(x) for x in A + B]))
    if grp: users = [u for u in users if A[0][u]['grp'] == grp]
    MA = np.array([[x[u]['n10'] for u in users] for x in A]); MB = np.array([[x[u]['n10'] for u in users] for x in B]); d = MB.mean(0) - MA.mean(0)
    rng = np.random.default_rng(BOOT_SEED); n = len(d); boot = d[rng.integers(0, n, (BOOT_N, n))].mean(1); lo, hi = np.percentile(boot, [2.5, 97.5])
    nz = d[np.abs(d) > 1e-15]; p = float(wilcoxon(nz).pvalue) if len(nz) > 10 else float('nan')
    return dict(n=n, delta=float(d.mean()), lo=float(lo), hi=float(hi), p=p, seeds_pos=int(((MB - MA).mean(1) > 0).sum()), sig=bool((lo > 0 or hi < 0) and p < 0.05),
                users_changed=int((np.abs(d) > 1e-15).sum()))


OUT = {'1_side_by_side': {}, '2_new_minus_paper': {}, '3_tg_vs_single': {}, '4_three_groups': {}, '5_best_epoch': {}}
for run, pu, lab in MODELS:
    P, N = res('paper', run), res('new', run)
    row = {}
    for sec in ('val', 'test', 'test_full_masked'):
        for g in ('overall', 'novel', 'repeat'):
            for nm, R in (('paper', P), ('new', N)):
                v = np.array([r[sec][g] for r in R]); row[f'{sec}|{g}|{nm}'] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), per_seed=v.tolist())
    OUT['1_side_by_side'][lab] = row
    OUT['5_best_epoch'][lab] = dict(paper=[r['best_epoch'] for r in P], new=[r['best_epoch'] for r in N],
                                    val_paper=[r['val']['overall'] for r in P], val_new=[r['val']['overall'] for r in N],
                                    elapsed_min_new=[r['elapsed_min'] for r in N])
    for p in PROTO:
        A, B = dumps('paper', pu, p), dumps('new', pu, p)
        OUT['2_new_minus_paper'][f'{lab}|{p}'] = {g: paired(A, B, None if g == 'all' else g) for g in ('all', 'novel', 'repeat')}
for p in PROTO:
    for src in ('paper', 'new'):
        TG = dumps(src, 'text_co_lgcn_late', p)
        for run, pu, lab in MODELS[1:]:
            OUT['3_tg_vs_single'][f'{p}|T+G - {lab}|{src}'] = {g: paired(dumps(src, pu, p), TG, None if g == 'all' else g) for g in ('all', 'novel', 'repeat')}

# 4: three target groups of Table 8
F, _ = feats('kc'); GROUPS = ['prev', 'linked', 'unlinked']; LAB = {'text': 'Text only', 'graph': 'Graph only', 'id': 'ID only'}
PUK = {'final': 'text_co_lgcn_late', 'text': 'text', 'graph': 'co_lgcn', 'id': 'id_fix'}
REF = json.load(open(AN + 'link_an_lg.json')) if os.path.exists(AN + 'link_an_lg.json') else None
tests, check = {}, []
for p in PROTO:
    for src in ('paper', 'new'):
        M = {}
        for k, pu in PUK.items():
            xs = dumps(src, pu, p); M[k] = {u: float(np.mean([x[u]['n10'] for x in xs])) for u in xs[0]}
        users = sorted(set.intersection(*[set(v) for v in M.values()]))
        for g in GROUPS:
            idx = [u for u in users if F[u]['rel'] == g]
            for a in LAB:
                r = test(np.array([M['final'][u] - M[a][u] for u in idx])); tests[f'{p}|{g}|T+G - {LAB[a]}|{src}'] = r
                if src == 'paper' and REF is not None:
                    ref = REF[f'kc|{p}|{g}|final-{a}']
                    check.append(dict(key=f'{p}|{g}|final-{a}', n=len(idx), n_ref=REF[f'kc|{p}|{g}|n'],
                                      max_abs_diff=max(abs(r[k] - ref[k]) for k in ('delta', 'lo', 'hi', 'p')), sig_same=r['sig'] == ref['sig']))
OUT['4_three_groups'] = dict(groups={g: int(sum(F[u]['rel'] == g for u in F)) for g in GROUPS}, tests=tests,
                             paper_check_vs_link_an_lg=None if not check else dict(max_abs_diff=max(c['max_abs_diff'] for c in check), all_sig_same=all(c['sig_same'] for c in check),
                                                                                   all_n_same=all(c['n'] == c['n_ref'] for c in check), rows=check))
json.dump(OUT, open(AN + 'valneg_check_an.json', 'w'), indent=1)
f = lambda r: f"{r['delta']:+.4f}{'*' if r['sig'] else ' '} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
for k, v in OUT['2_new_minus_paper'].items(): print('new - paper', k, f(v['all']))
for k, v in OUT['4_three_groups']['tests'].items(): print(k, f(v))
