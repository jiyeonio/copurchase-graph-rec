"""Every table and every number of the paper from the result files: writes <PAPER>/tables/T_*.tex and <PAPER>/numbers.tex (macros).
Inputs: result files of the runs, per-user dumps, the analysis files of analysis/*.py and the evaluation of the baselines (09).
The numbers of Park & Oh (2024) quoted in the text are the constants below. Anything missing is rendered as [TODO]."""
import json, glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import *
from scipy.stats import wilcoxon
U, BASE, OUT = RUNS, BASELINES, PAPER; os.makedirs(OUT + 'tables', exist_ok=True)
RUNS = {'id': ('kc11/K_id', 'ic11/K_id'), 'text': ('kc11/I_text', 'ic11/I_text'), 'graph': ('kc11/I_co_lgcn', 'ic11/I_co_lgcn'),
        'final': ('kc11/I_text_co_lgcn_late', 'ic11/I_text_co_lgcn_late')}
PUNAME = {'id': 'id_fix', 'text': 'text', 'graph': 'co_lgcn', 'final': 'text_co_lgcn_late'}
LABEL = {'id': 'ID', 'text': 'Text', 'graph': 'Graph', 'final': 'Text+Graph'}
DSI = {'kc': 0, 'ic': 1}
BNAMES = (('LightGCN', 'lightgcn'), ('BERT4Rec', 'bert4rec'), ('GRU4Rec', 'gru4rec'), ('SASRec', 'sasrec'))
# Park & Oh (2024), Katcher's, as printed there (NDCG@10, NDCG@5)
PARKOH = (('DeepFM (with concurrent purchases)', 0.3423, 0.2767), ('RLMRec (LightGCN)', 0.3623, 0.3132), ('XGBoost (with concurrent purchases)', 0.4333, 0.3815),
          ('BERT (without concurrent purchases)', 0.5037, 0.4623), ('BERT (with concurrent purchases)', 0.5376, 0.4910))

macros = {}
def mac(name, val): macros[name] = val
def pm(m, s): return f'{m:.4f} $\\pm$ {s:.4f}'
def load(key, ds):
    fs = sorted(glob.glob(f'{U}{RUNS[key][DSI[ds]]}/result_U_seed*.json')); return [json.load(open(f)) for f in fs] if len(fs) == 5 else None
def vals(rs, k, s='overall'): return np.array([r[k][s] for r in rs], dtype=float)
def ms(rs, k, with_pm=True):
    if rs is None: return '[TODO]'
    v = vals(rs, k); return pm(v.mean(), v.std(ddof=1)) if with_pm else f'{v.mean():.4f}'
def bload(ds, m):
    p = f'{BASE}{ds}/{m}/full_eval.json'
    if not os.path.exists(p): return None
    d = json.load(open(p)); return d['per_seed'] if isinstance(d, dict) and len(d.get('per_seed', [])) == 5 else None
def bms(rs, f, with_pm=True):
    if rs is None: return '[TODO]'
    v = np.array([f(r) for r in rs], dtype=float); return pm(v.mean(), v.std(ddof=1)) if with_pm else f'{v.mean():.4f}'

# ----------------------------------------------------------------- paired tests
_PUC = {}
def pu_load(ds, key, proto):
    if (ds, key, proto) in _PUC: return _PUC[(ds, key, proto)]
    r = None
    if key.startswith('base:'):
        fs = [f'{BASE}{ds}/{key[5:]}/per_user_seed{s}.json' for s in SEEDS]
        if all(os.path.exists(f) for f in fs) and bload(ds, key[5:]) is not None:
            r = []
            for f in fs:
                d = json.load(open(f)); src = d['1to49'] if proto == '1to49' else d['full']; kk = 'ndcg@10' if proto == '1to49' else 'masked@10'
                r.append({u: x[kk] for u, x in src.items()})
    else:
        fs = [f'{PU}{ds}_{PUNAME[key]}_{proto}_seed{s}.json' for s in SEEDS]
        if all(os.path.exists(f) for f in fs): r = [{u: x['n10'] for u, x in json.load(open(f)).items()} for f in fs]
    _PUC[(ds, key, proto)] = r; return r
def paired(ds, a, b, proto):
    """b - a on seed-averaged per-user NDCG@10; bootstrap over users, percentile 95% CI; two-sided Wilcoxon, zeros excluded"""
    A, B = pu_load(ds, a, proto), pu_load(ds, b, proto)
    if A is None or B is None: return None
    users = sorted(set.intersection(*[set(x) for x in A], *[set(x) for x in B]))
    MA = np.array([[x[u] for u in users] for x in A]); MB = np.array([[x[u] for u in users] for x in B]); d = MB.mean(0) - MA.mean(0)
    rng = np.random.default_rng(BOOT_SEED); n = len(d); boot = d[rng.integers(0, n, (BOOT_N, n))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5]); nz = d[np.abs(d) > 1e-15]; p = float(wilcoxon(nz).pvalue)
    return dict(n=n, delta=float(d.mean()), lo=float(lo), hi=float(hi), p=p, seeds_pos=int(((MB - MA).mean(1) > 0).sum()), sig=bool((lo > 0 or hi < 0) and p < 0.05))
def fmt_pair(r):
    if r is None: return '[TODO] & [TODO] & [TODO]'
    return f"{r['delta']:+.4f}{'$^{*}$' if r['sig'] else ''} [{r['lo']:+.4f}, {r['hi']:+.4f}] & " + ('$<$0.001' if r['p'] < 0.001 else f"{r['p']:.3f}") + f" & {r['seeds_pos']}/5"

W = {(ds, k): load(k, ds) for ds in ('kc', 'ic') for k in RUNS}
for (ds, key), rs in W.items():
    for tag, k in (('A', 'test'), ('Afive', 'test_ndcg5'), ('F', 'test_full_masked'), ('Ffive', 'test_full_masked_ndcg5')):
        mac(f'{ds}{key}{tag}', ms(rs, k)); mac(f'{ds}{key}{tag}m', ms(rs, k, False))
    for v in ('text', 'graph'):
        mac(f'{ds}{key}w{v}', '[TODO]' if rs is None else f"{np.mean([r['view_weights'].get(v, 0) for r in rs]):.2f}")
BF = {'F': lambda r: r['full_masked_ndcg@10_overall'], 'Ffive': lambda r: r['full_masked_ndcg@5_overall'],
      'A': lambda r: r['test_1to49']['overall']['ndcg@10'], 'Afive': lambda r: r['test_1to49']['overall']['ndcg@5']}
for ds in ('kc', 'ic'):
    for bn, bm in BNAMES:
        rs = bload(ds, bm); nm = bn.replace('4', 'four')
        for tag, f in BF.items(): mac(f'{ds}base{nm}{tag}', bms(rs, f)); mac(f'{ds}base{nm}{tag}m', bms(rs, f, False))

# ----------------------------------------------------------------- Table: paired tests, Text+Graph against each single representation
L = [r'\begin{tabular}{lllccc}', r'\toprule', r'Dataset & Comparison & Protocol & $\Delta$NDCG@10 [95\% CI] & $p$ & Seeds \\', r'\midrule']
for ds, dn in (('kc', "Katcher's"), ('ic', 'Instacart')):
    first = True
    for a in ('id', 'text', 'graph'):
        for proto, pn, P_ in (('1to49', '1:49', 'A'), ('full', 'Full catalogue', 'F')):
            r = paired(ds, a, 'final', proto)
            L.append(f"{dn if first else ''} & Text+Graph $-$ {LABEL[a]} & {pn} & {fmt_pair(r)} \\\\"); first = False
            mac(f'{ds}d{a}{P_}', '[TODO]' if r is None else f"{r['delta']:+.4f}"); mac(f'{ds}sig{a}{P_}', '[TODO]' if r is None else ('yes' if r['sig'] else 'no'))
    if ds == 'kc': L.append(r'\midrule')
L += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}tables/T_sig.tex', 'w').write('\n'.join(L) + '\n')
# Text+Graph against each baseline (macros only; full catalogue for both datasets, 1:49 where the evaluated checkpoints exist)
for ds in ('kc', 'ic'):
    for bn, bm in BNAMES:
        for proto, P_ in (('1to49', 'A'), ('full', 'F')):
            r = paired(ds, 'base:' + bm, 'final', proto); nm = bn.replace('4', 'four')
            mac(f'{ds}dvs{nm}{P_}', '[TODO]' if r is None else f"{r['delta']:+.4f}"); mac(f'{ds}sigvs{nm}{P_}', '[TODO]' if r is None else ('yes' if r['sig'] else 'no'))
            mac(f'{ds}civs{nm}{P_}', '[TODO]' if r is None else f"[{r['lo']:+.4f}, {r['hi']:+.4f}]")

# ----------------------------------------------------------------- Strata (analysis/link_an_lg.py)
LA = json.load(open(AN + 'link_an_lg.json'))
def dl(r, ci=True):
    if r is None: return '--'
    return f"{r['delta']:+.4f}{'$^{*}$' if r['sig'] else ''}" + (f" [{r['lo']:+.4f}, {r['hi']:+.4f}]" if ci else '')
GL = {'prev': 'purchased before', 'linked': 'new, with co-purchase link to history', 'unlinked': 'new, no co-purchase link'}
NUMW = {'kc': 'Katcher\'s', 'ic': 'Instacart'}
# Table: Text+Graph against Text by the relation between the target and the history
L = [r'\begin{tabular}{llrcccccc}', r'\toprule', r' & & & \multicolumn{3}{c}{1:49} & \multicolumn{3}{c}{Full catalogue} \\', r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}',
     r'Dataset & Target item & Users & Text & Text+Graph & $\Delta$ & Text & Text+Graph & $\Delta$ \\', r'\midrule']
for ds in ('kc', 'ic'):
    first = True
    for g in ('prev', 'linked', 'unlinked'):
        n = LA[f'{ds}|1to49|{g}|n']
        if f'{ds}|1to49|{g}|text' not in LA: continue                      # fewer than 30 users (Instacart, unlinked: 26)
        cells = []
        for proto in ('1to49', 'full'):
            cells += [f"{LA[f'{ds}|{proto}|{g}|text']:.4f}", f"{LA[f'{ds}|{proto}|{g}|final']:.4f}", dl(LA[f'{ds}|{proto}|{g}|final-text'], False)]
        L.append(f"{NUMW[ds] if first else ''} & {GL[g]} & {n:,} & " + ' & '.join(cells) + r' \\'); first = False
        for proto, P_ in (('1to49', 'A'), ('full', 'F')):
            for pair, nm in (('final-text', 'ft'), ('final-final_rw', 'frw'), ('final_rw-text', 'rwt')):
                r = LA[f'{ds}|{proto}|{g}|{pair}']; mac(f'{ds}link{g}{nm}{P_}', f"{r['delta']:+.4f}" + ('$^{*}$' if r['sig'] else ''))
        mac(f'{ds}link{g}n', f'{n:,}')
    if ds == 'kc': L.append(r'\midrule')
L += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}tables/T_link.tex', 'w').write('\n'.join(L) + '\n')
# Table: Text+Graph against Graph by the co-purchase degree of the target
L = [r'\begin{tabular}{llrcccccc}', r'\toprule', r' & & & \multicolumn{3}{c}{1:49} & \multicolumn{3}{c}{Full catalogue} \\', r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}',
     r'Dataset & Degree of target & Users & Graph & Text+Graph & $\Delta$ & Graph & Text+Graph & $\Delta$ \\', r'\midrule']
for ds in ('kc', 'ic'):
    q = LA[f'{ds}|1to49|deg_cuts']; lo, hi = int(np.floor(q[0])), int(np.floor(q[1])); first = True
    for g, lab in (('deg0', '0 (no edge)'), ('deglow', f'1--{lo:,}'), ('degmid', f'{lo + 1:,}--{hi:,}'), ('deghigh', f'$>${hi:,}')):
        if f'{ds}|1to49|{g}|graph' not in LA: continue
        cells = []
        for proto in ('1to49', 'full'):
            cells += [f"{LA[f'{ds}|{proto}|{g}|graph']:.4f}", f"{LA[f'{ds}|{proto}|{g}|final']:.4f}", dl(LA[f'{ds}|{proto}|{g}|final-graph'], False)]
        L.append(f"{NUMW[ds] if first else ''} & {lab} & {LA[f'{ds}|1to49|{g}|n']:,} & " + ' & '.join(cells) + r' \\'); first = False
        for proto, P_ in (('1to49', 'A'), ('full', 'F')):
            r = LA[f'{ds}|{proto}|{g}|final-graph']; mac(f'{ds}{g.replace("0", "zero")}fg{P_}', f"{r['delta']:+.4f}" + ('$^{*}$' if r['sig'] else ''))
    if ds == 'kc': L.append(r'\midrule')
L += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}tables/T_degree.tex', 'w').write('\n'.join(L) + '\n')
# Graph control: the real co-purchase graph against a degree-preserving rewiring of it (LightGCN on both). * = real and rewired differ significantly.
PN = {'1to49': '1:49', 'full': 'Full catalogue', 'unmasked': 'Full catalogue, no item removed'}
SUF = {'1to49': 'A', 'full': 'F', 'unmasked': 'U'}
CTLTESTS = {}
def ctl3(ds, proto, g, base):
    rw = 'final_rw' if base == 'final' else 'graph_rw'; r = LA[f'{ds}|{proto}|{g}|{base}-{rw}']
    return [f"{LA[f'{ds}|{proto}|{g}|{base}']:.4f}", f"{LA[f'{ds}|{proto}|{g}|{rw}']:.4f}", f"{r['delta']:+.4f}" + ('$^{*}$' if r['sig'] else '')]
for base, pre in (('graph', 'gctl'), ('final', 'ctl')):
    for ds in ('kc', 'ic'):
        for proto in ('1to49', 'full'):
            for g in ('linked', 'all', 'prev'):
                c = ctl3(ds, proto, g, base); mac(f'{ds}{pre}{g}real{SUF[proto]}', c[0]); mac(f'{ds}{pre}{g}rw{SUF[proto]}', c[1]); mac(f'{ds}{pre}d{g}rw{SUF[proto]}', c[2])
                if g != 'prev': CTLTESTS[f"{'TG' if base == 'graph' else 'TT'} {ds} {proto} {g} {base} real-rewired"] = LA[f"{ds}|{proto}|{g}|{base}-{'final_rw' if base == 'final' else 'graph_rw'}"]
L = [r'\begin{tabular}{lllrcccccc}', r'\toprule', r' & & & & \multicolumn{3}{c}{Graph only} & \multicolumn{3}{c}{Text+Graph} \\', r'\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
     r'Dataset & Protocol & Target item & Users & Real & Rewired & $\Delta$ & Real & Rewired & $\Delta$ \\', r'\midrule']
for ds in ('kc', 'ic'):
    for proto in ('1to49', 'full'):
        first = True
        for g in ('prev', 'linked', 'unlinked', 'all'):
            if f'{ds}|{proto}|{g}|text' not in LA: continue
            L.append(f"{NUMW[ds] if (first and proto == '1to49') else ''} & {PN[proto] if first else ''} & {GL.get(g, 'all users')} & {LA[f'{ds}|{proto}|{g}|n']:,} & " + ' & '.join(ctl3(ds, proto, g, 'graph') + ctl3(ds, proto, g, 'final')) + r' \\'); first = False
        if not (ds == 'ic' and proto == 'full'): L.append(r'\cmidrule(lr){2-10}' if proto == '1to49' else r'\midrule')
L += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}tables/T_control_full.tex', 'w').write('\n'.join(L) + '\n')
for ds in ('kc', 'ic'): mac(f'{ds}isolated', f"{100 * LA[f'{ds}|isolated_items']:.1f}\\%")
# trained parameters of the proposed model (millions), counted in the seed-42 checkpoint on Katcher's; the count is the same on Instacart
import torch as _torch
_sd = _torch.load(f"{U}{RUNS['final'][0]}/best_base_real_seed42.pt", map_location='cpu')
mac('oursparams', f"{sum(v.numel() for v in _sd.values()) / 1e6:.2f}")
# ----------------------------------------------------------------- Appendix table: full catalogue without removing history items
L = [r'\begin{tabular}{lcc}', r'\toprule', r"Model & Katcher's & Instacart \\", r'\midrule']
for bn, bm in BNAMES: L.append(f'{bn} & ' + ' & '.join(bms(bload(ds, bm), lambda r: r['full_unmasked_ndcg@10_overall']) for ds in ('kc', 'ic')) + r' \\')
L.append(r'\midrule')
for key in ('id', 'text', 'graph', 'final'):
    lab = r'\textbf{Text+Graph (proposed)}' if key == 'final' else LABEL[key] + ' only'
    L.append(f'{lab} & ' + ' & '.join(ms(W[(ds, key)], 'test_full_unmasked') for ds in ('kc', 'ic')) + r' \\')
L += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}tables/T_unmasked.tex', 'w').write('\n'.join(L) + '\n')

# ----------------------------------------------------------------- Appendix: co-purchase threshold on Katcher's (graphs gf2/gf3, runs kc11/S_*)
FS = {1: ('kc11/I_co_lgcn', 'kc11/I_text_co_lgcn_late', GRAPHS + 'g1/stats.json'),
      2: ('kc11/S_co_f2', 'kc11/S_text_co_f2_late', GRAPHS + 'gf2/stats.json'),
      3: ('kc11/S_co_f3', 'kc11/S_text_co_f3_late', GRAPHS + 'gf3/stats.json')}
def ldir(d):
    fs = sorted(glob.glob(f'{U}{d}/result_U_seed*.json')); return [json.load(open(f)) for f in fs] if len(fs) == 5 else None
L = [r'\begin{tabular}{crccccc}', r'\toprule', r' & & & \multicolumn{2}{c}{Graph only} & \multicolumn{2}{c}{Text+Graph} \\', r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
     r'$f$ & Edges & Items without edge & 1:49 & Full catalogue & 1:49 & Full catalogue \\', r'\midrule']
for f_, (g, tg_, st) in FS.items():
    st = json.load(open(st)); rg, rt = ldir(g), ldir(tg_)
    L.append(f"$\\ge{f_}$ & {st['edges']:,} & {100 * st['isolated_ratio']:.1f}\\% & {ms(rg, 'test')} & {ms(rg, 'test_full_masked')} & {ms(rt, 'test')} & {ms(rt, 'test_full_masked')} \\\\")
    mac(f'kcfs{["", "one", "two", "three"][f_]}graphA', ms(rg, 'test', False)); mac(f'kcfs{["", "one", "two", "three"][f_]}finalF', ms(rt, 'test_full_masked', False)); mac(f'kcfs{["", "one", "two", "three"][f_]}graphF', ms(rg, 'test_full_masked', False))
L += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}tables/T_fsens.tex', 'w').write('\n'.join(L) + '\n')

# ----------------------------------------------------------------- Encoder and training variants (runs {kc11,ic11}/T_*, paired tests analysis/ablation_an.py)
AB = json.load(open(AN + 'ablation_an.json'))
for v in ('itempos_causal', 'nodense'):
    for ds in ('kc', 'ic'):
        rs = ldir(f'{ds}11/T_{v}')
        for proto, k, P_ in (('1to49', 'test', 'A'), ('full', 'test_full_masked', 'F')):
            a = AB[f'{ds}|{proto}|{v}']; star = r'$^{*}$' if a['sig'] else ''
            nm = 'itemseq' if v == 'itempos_causal' else 'lastonly'
            mac(f'{ds}abl{nm}{P_}', f'{vals(rs, k).mean():.4f}'); mac(f'{ds}abl{nm}d{P_}', f"{-a['delta']:+.4f}" + star)

# ----------------------------------------------------------------- Nearest neighbours of the two fixed representations (analysis/emb_vis2.py -> tables/emb_knn.json)
KN = json.load(open(f'{OUT}tables/emb_knn.json'))
L = [r'\begin{tabular}{llccc}', r'\toprule', r'Dataset & Among the 10 nearest neighbours: share of items & Text & Graph & Random \\', r'\midrule']
for dn, ds, rows in (("Katcher's", 'kc', (('cat', 'in the same top-level category'), ('fine', 'in the same leaf category'), ('link', 'with a co-purchase edge to the item'))),
                     ('Instacart', 'ic', (('cat', 'in the same aisle'), ('link', 'with a co-purchase edge to the item')))):
    for i, (k, lab) in enumerate(rows):
        L.append(f"{dn if i == 0 else ''} & {lab} & {KN[dn]['text'][k]:.3f} & {KN[dn]['graph'][k]:.3f} & {KN[dn]['random'][k]:.3f} \\\\")
        for m in ('text', 'graph', 'random'): mac(f'{ds}knn{k}{m}', f"{KN[dn][m][k]:.2f}")
    if ds == 'kc': L.append(r'\midrule')
L += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}tables/T_knn.tex', 'w').write('\n'.join(L) + '\n')
# ----------------------------------------------------------------- Appendix table: Text vs Text+Graph by target group when no item is removed
L = [r'\begin{tabular}{llrccc}', r'\toprule', r'Dataset & Target item & Users & Text & Text+Graph & $\Delta$ \\', r'\midrule']
for ds in ('kc', 'ic'):
    first = True
    for g in ('prev', 'linked', 'unlinked', 'all'):
        if f'{ds}|unmasked|{g}|text' not in LA: continue
        L.append(f"{NUMW[ds] if first else ''} & {GL.get(g, 'all users')} & {LA[f'{ds}|unmasked|{g}|n']:,} & {LA[f'{ds}|unmasked|{g}|text']:.4f} & {LA[f'{ds}|unmasked|{g}|final']:.4f} & {dl(LA[f'{ds}|unmasked|{g}|final-text'], False)} \\\\"); first = False
    if ds == 'kc': L.append(r'\midrule')
L += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}tables/T_link_unmasked.tex', 'w').write('\n'.join(L) + '\n')
mac('iclinklinkedtextU', f"{LA['ic|unmasked|linked|text']:.4f}"); mac('iclinklinkedfinalU', f"{LA['ic|unmasked|linked|final']:.4f}")

# ----------------------------------------------------------------- Additional statements (analysis/extra_an2.py, extra_ordsize.py)
EX = EXTRA; A2 = json.load(open(EX + 'an2.json')); OS_ = json.load(open(EX + 'ordsize.json'))
# proposed model minus the item-level variant, Katcher's test users by the mean number of items per order of their history (= 1 / at least 1.5)
for lab, nm in (('1.0', 'single'), ('>1.5', 'multi')):
    for proto, P_ in (('1to49', 'A'), ('full', 'F')):
        r = A2[f'kc|C|{proto}|meansize|{lab}|itempos_causal']; mac(f'kcenc{nm}{P_}', f"{r['delta']:+.4f}" + ('$^{*}$' if r['sig'] else ''))
    mac(f'kcenc{nm}n', f"{A2[f'kc|C|1to49|meansize|{lab}|itempos_causal']['n']:,}")
assert A2['kc|C|1to49|meansize|1.0|itempos_causal']['n'] == OS_['kc']['single'] and A2['kc|C|1to49|meansize|>1.5|itempos_causal']['n'] == OS_['kc']['atleast15']
mac('icencbelown', f"{OS_['ic']['below15']:,}")
# gain of Text+Graph over Graph within popularity terciles of the target, lower / upper half by degree (Katcher's, 1:49); rank correlation of degree and popularity
lo_ = [A2[f'kc|E|1to49|pop{i}|lowdeg']['gain'] for i in range(3)]; hi_ = [A2[f'kc|E|1to49|pop{i}|highdeg']['gain'] for i in range(3)]
mac('kcpoplowmin', f'{min(lo_):.3f}'); mac('kcpoplowmax', f'{max(lo_):.3f}'); mac('kcpophighmin', f'{min(hi_):.3f}'); mac('kcpophighmax', f'{max(hi_):.3f}')
mac('kcdegpoprho', f"{A2['kc|E|1to49|spearman_deg_pop']:.2f}"); mac('icdegpoprho', f"{A2['ic|E|1to49|spearman_deg_pop']:.2f}")
# Holm correction over the paired tests of the tables of the main text (sig, link, both control tables, ablation, degree) -> tables/holm.json
HT = {}
for ds in ('kc', 'ic'):
    for a in ('id', 'text', 'graph'):
        for proto in ('1to49', 'full'): HT[f'S {ds} {proto} T+G-{a}'] = paired(ds, a, 'final', proto)
    for proto in ('1to49', 'full'):
        for g in ('prev', 'linked', 'unlinked'):
            if f'{ds}|{proto}|{g}|final-text' in LA and f'{ds}|{proto}|{g}|text' in LA: HT[f'L {ds} {proto} {g} T+G-Text'] = LA[f'{ds}|{proto}|{g}|final-text']
        for g in ('deg0', 'deglow', 'degmid', 'deghigh'):
            if f'{ds}|{proto}|{g}|graph' in LA: HT[f'D {ds} {proto} {g} T+G-Graph'] = LA[f'{ds}|{proto}|{g}|final-graph']
        HT[f'A {ds} {proto} final-itempos_causal'] = AB[f'{ds}|{proto}|itempos_causal']
HT.update(CTLTESTS)
_o = sorted(HT, key=lambda k: HT[k]['p']); _m = len(_o); _rej = True; HROWS = []
for _r, _k in enumerate(_o):
    _rej = _rej and HT[_k]['p'] <= 0.05 / (_m - _r); HROWS.append(dict(test=_k, p=HT[_k]['p'], delta=HT[_k]['delta'], sig=bool(HT[_k]['sig']), holm=bool(_rej and HT[_k]['sig'])))
json.dump(dict(n_tests=_m, sig_before=sum(r['sig'] for r in HROWS), sig_after=sum(r['holm'] for r in HROWS), rows=HROWS), open(f'{OUT}tables/holm.json', 'w'), indent=1)
mac('holmlost', str(sum(r['sig'] for r in HROWS) - sum(r['holm'] for r in HROWS))); mac('holmtests', str(_m)); mac('holmbefore', str(sum(r['sig'] for r in HROWS))); mac('holmafter', str(sum(r['holm'] for r in HROWS)))

# ----------------------------------------------------------------- (a) linked new targets by recency and strength of the link (analysis/story_an.py)
ST_ = json.load(open(EX + 'story_new.json'))
def sdl(r): return f"{r['delta']:+.4f}" + ('$^{*}$' if r['sig'] else '')
ROWS_ = (('A', 'linked', 'all'), ('B', 'last', 'last'), ('B', 'earlier', 'earlier'), ('B', 'weak', 'weak'), ('B', 'medium', 'medium'), ('B', 'strong', 'strong'))
for blk, g, nm in ROWS_:
    for proto, P_ in (('1to49', 'A'), ('full', 'F')):
        o = ST_[f'{blk}|{proto}|{g}']
        for k, kn in (('real_text', 'ft'), ('real_rw', 'frw')): mac(f'kcst{nm}{kn}{P_}', sdl(o[k]))
    mac(f'kcst{nm}n', f"{ST_[f'{blk}|1to49|{g}']['n']:,}")
assert ST_['A|1to49|linked']['n'] == LA['kc|1to49|linked|n'] and all(abs(ST_[f'A|{p}|linked']['real_text']['delta'] - LA[f'kc|{p}|linked|final-text']['delta']) < 1e-12 for p in ('1to49', 'full'))
# graph gain (Text+Graph - Text) by the mean number of items per order of the history
for lab, nm in (('1.0', 'single'), ('>1.5', 'multi')):
    for proto, P_ in (('1to49', 'A'), ('full', 'F')): mac(f'kcgsz{nm}{P_}', sdl(ST_[f'C|{proto}|meansize|{lab}']['real_text']))
    assert ST_[f'C|1to49|meansize|{lab}']['n'] == A2[f'kc|C|1to49|meansize|{lab}|itempos_causal']['n']
# (b) removal of co-purchase edges (graphs/drop_edges.py, analysis/edge_removal_an.py); groups by the full graph
B23 = json.load(open(EX + 'boost23.json'))
L = [r'\begin{tabular}{llrccccc}', r'\toprule', r' & & & \multicolumn{4}{c}{Share of co-purchase edges kept} & \\', r'\cmidrule(lr){4-7}',
     r'Protocol & Test users & Users & 100\% & 75\% & 50\% & 25\% & 100\% $-$ 25\% \\', r'\midrule']
for proto, P_ in (('1to49', 'A'), ('full', 'F')):
    for i, (g, lab, gn) in enumerate((('linked', 'new item linked to the history', 'linked'), ('all', 'all users', 'all'))):
        o = B23[f'E2|{proto}|{g}']
        L.append(f"{PN[proto] if i == 0 else ''} & {lab} & {o['n']:,} & " + ' & '.join(sdl(o[f'gain{k}']) for k in (100, 75, 50, 25)) + f" & {sdl(o['d100_25'])} \\\\")
        for k in (100, 75, 50, 25): mac(f'kcedge{gn}{["", "q", "h", "t"][(100 - k) // 25] if k < 100 else "full"}{P_}', sdl(o[f'gain{k}']))
        mac(f'kcedge{gn}diff{P_}', sdl(o['d100_25']))
    if proto == '1to49': L.append(r'\midrule')
L += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}tables/T_edges.tex', 'w').write('\n'.join(L) + '\n')
assert all(abs(B23[f'E2|{p}|{g}']['gain100']['delta'] - LA[f'kc|{p}|{g}|final-text']['delta']) < 1e-12 for p in ('1to49', 'full') for g in ('linked', 'all'))
ED_ = {d: sum(1 for _ in open(f'{GRAPHS}gdrop{d}/graph_edges.csv')) - 1 for d in (25, 50, 75)}
for d, nm in ((25, 'q'), (50, 'h'), (75, 't')): mac(f'kcedgen{nm}', f'{ED_[d]:,}')
# (c) appendix comparison: Instacart with an English encoder of the same family (google/electra-base-discriminator; runs ic11/Z_*_electra)
UO, PU3 = RUNS_ENC, PU_ENC
def _runs(d):
    fs = sorted(glob.glob(f'{UO}{d}/result_U_seed*.json')); assert len(fs) == 5, d; return [json.load(open(f)) for f in fs]
def _pu3(name, proto): return [{u: x['n10'] for u, x in json.load(open(f'{PU3}ic_{name}_{proto}_seed{s}.json')).items()} for s in SEEDS]
def _pair(A, B):
    users = sorted(set.intersection(*[set(x) for x in A], *[set(x) for x in B])); d = np.array([[x[u] for u in users] for x in B]).mean(0) - np.array([[x[u] for u in users] for x in A]).mean(0)
    rng = np.random.default_rng(BOOT_SEED); n = len(d); boot = d[rng.integers(0, n, (BOOT_N, n))].mean(1); lo, hi = np.percentile(boot, [2.5, 97.5]); nz = d[np.abs(d) > 1e-15]; p = float(wilcoxon(nz).pvalue)
    return dict(n=n, delta=float(d.mean()), lo=float(lo), hi=float(hi), p=p, sig=bool((lo > 0 or hi < 0) and p < 0.05))
EL = {'text': _runs('ic11/Z_text_electra'), 'final': _runs('ic11/Z_text_co_lgcn_late_electra')}; ELP = {'text': 'text_electra', 'final': 'text_co_lgcn_late_electra'}
L = [r'\begin{tabular}{llcccc}', r'\toprule', r' & & \multicolumn{2}{c}{1:49} & \multicolumn{2}{c}{Full catalogue} \\', r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
     r'Model & Text encoder & NDCG@10 & $\Delta$ to ID & NDCG@10 & $\Delta$ to ID \\', r'\midrule',
     'ID only & -- & ' + ms(W[('ic', 'id')], 'test') + ' & -- & ' + ms(W[('ic', 'id')], 'test_full_masked') + r' & -- \\', r'\midrule']
for key, lab in (('text', 'Text only'), ('final', 'Text+Graph')):
    for enc, en in (('kr', 'KR-ELECTRA'), ('en', 'ELECTRA-base')):
        cells = []
        for proto, k, P_ in (('1to49', 'test', 'A'), ('full', 'test_full_masked', 'F')):
            rs = W[('ic', key)] if enc == 'kr' else EL[key]; mine = pu_load('ic', key, proto) if enc == 'kr' else _pu3(ELP[key], proto)
            r = _pair(pu_load('ic', 'id', proto), mine); cells += [ms(rs, k), sdl(r)]
            mac(f'icenc{enc}{key}{P_}', ms(rs, k, False)); mac(f'icenc{enc}{key}did{P_}', sdl(r))
            if enc == 'en':
                r2 = _pair(pu_load('ic', key, proto), mine); mac(f'icencgain{key}{P_}', sdl(r2))          # ELECTRA-base minus KR-ELECTRA, same model
            if key == 'final':
                r3 = _pair(pu_load('ic', 'text', proto) if enc == 'kr' else _pu3(ELP['text'], proto), mine); mac(f'icenc{enc}ft{P_}', sdl(r3))   # Text+Graph minus Text, same encoder
        L.append(f"{lab if enc == 'kr' else ''} & {en} & " + ' & '.join(cells) + r' \\')
    if key == 'text': L.append(r'\midrule')
L += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}tables/T_ic_enc.tex', 'w').write('\n'.join(L) + '\n')
TK = json.load(open(EXTRA_ENC + 'tok_stats.json'))
for enc, k in (('kr', 'KR-ELECTRA'), ('en', 'ELECTRA-base (en)')): mac(f'ictok{enc}whole', f"{100 * TK[k]['words_whole_share']:.1f}\\%"); mac(f'ictok{enc}len', f"{TK[k]['tokens_mean']:.1f}")
# a few plain macros for the text
mac('kcablitemseqsigA', 'yes' if AB['kc|1to49|itempos_causal']['sig'] else 'no')
for g in ('prev', 'linked', 'unlinked'): mac(f'kcshare{g}', f"{100 * LA[f'kc|1to49|{g}|n'] / LA['kc|1to49|all|n']:.0f}\\%")
mac('iclinkprevftA', sdl(LA['ic|1to49|prev|final-text'])); mac('iclinkprevftF', sdl(LA['ic|full|prev|final-text']))
for a in ('id', 'text'):
    for proto, P_ in (('1to49', 'A'), ('full', 'F')):
        for ds in ('kc', 'ic'):
            r = paired(ds, a, 'final', proto); mac(f'{ds}ds{a}{P_}', sdl(r))
for proto, P_ in (('1to49', 'A'), ('full', 'F')): mac(f'kclinkedgain{P_}', f"{LA[f'kc|{proto}|linked|final-text']['delta']:+.4f}"); mac(f'iclinkedgain{P_}', f"{LA[f'ic|{proto}|linked|final-text']['delta']:+.4f}")
mac('kcsharenew', f"{100 * (1 - LA['kc|1to49|prev|n'] / LA['kc|1to49|all|n']):.0f}\\%"); mac('icshareprev', f"{100 * LA['ic|1to49|prev|n'] / LA['ic|1to49|all|n']:.0f}\\%")
mac('kcstcutlo', f"{int(ST_['B|1to49|weak']['cuts'][0])}"); mac('kcstcuthi', f"{int(ST_['B|1to49|weak']['cuts'][1])}")      # cut points of the link strength (integers, ties at the cuts)
assert ST_['B|1to49|weak']['cuts'] == [float(int(c)) for c in ST_['B|1to49|weak']['cuts']]
mac('kcstlastgainA', f"{ST_['B|1to49|last']['real_text']['delta']:+.4f}"); mac('kcstlastgainF', f"{ST_['B|full|last']['real_text']['delta']:+.4f}")
# constants used in the prose (so that no number is typed by hand there)
mac('parkohbest', f'{PARKOH[-1][1]:.4f}'); mac('parkohbase', f'{PARKOH[-2][1]:.4f}'); mac('topk', '10')
_CS = json.load(open(f'{OUT}tables/check_stats.json'))
mac('kcordersperuser', f"{_CS['kc_train_data']['orders_per_user_mean']:.2f}"); mac('icordersperuser', f"{_CS['ic_train']['orders_per_user_mean']:.1f}"); mac('icitemsperorder', f"{_CS['ic_train']['items_per_order_mean']:.2f}")
mac('iclinkedhi', f"{np.ceil(100 * max(LA[f'ic|full|linked|{m}'] for m in ('text', 'final', 'final_rw', 'graph', 'graph_rw'))) / 100:.2f}")

import re as _re
def _minus(t): return _re.sub(r'(?<![\w$\\{-])-(?=\d)', r'\\ensuremath{-}', t)       # a hyphen before a digit is a minus sign: typeset it as one (text or math mode)
for _f in glob.glob(f'{OUT}tables/T_*.tex'):
    _t = open(_f).read(); open(_f, 'w').write(_minus(_t))
with open(f'{OUT}numbers.tex', 'w') as f:
    for k, v in macros.items():
        f.write(f'\\newcommand{{\\{k}}}{{{_minus(v)}}}\n')
print('macros:', len(macros), '| TODO macros:', sum('[TODO]' in v for v in macros.values()))
