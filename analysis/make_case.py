"""Case-study figure: two real Instacart test users (seed 42 checkpoint of the proposed model). For each user the last three input orders,
the held-out next purchase, and the five highest-ranked items of the text representation, the graph representation and their combination
(full catalogue, items of the history removed). Source: <AN>/case_ic_seed42.json (--dump-case of model_instacart.py). -> <PAPER>/figs/fig_case.tex"""
import json
from common import *
im = json.load(open(IC + 'item_meta.json'))
D = json.load(open(AN + 'case_ic_seed42.json'))['users']
USERS = [('197342', 'User A'), ('36963', 'User B')]
def nm(i): return im[str(i)]['text'].split(' - ')[-1].replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
def orders(e):
    o = {}
    for it, t in zip(e['history'], e['tx']): o.setdefault(t, []).append(it)
    return [o[k] for k in sorted(o)]
L = [r'\footnotesize', r'\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}']
for u, lab in USERS:
    e = D[u]; od = orders(e); n = len(od)
    hit = lambda i: (r'\textbf{' + nm(i) + r'} $\checkmark$') if i == e['target'] else nm(i)
    L += [r'\begin{tabular}{@{}p{0.31\linewidth}p{0.31\linewidth}p{0.31\linewidth}@{}}', r'\toprule',
          r'\multicolumn{3}{@{}p{0.96\linewidth}@{}}{\textbf{' + lab + f'}}: last three of {n} orders: ' + '; '.join(r'\{' + ', '.join(nm(i) for i in o) + r'\}' for o in od[-3:]) + r'} \\',
          r'\multicolumn{3}{@{}p{0.96\linewidth}@{}}{Next purchase: \textbf{' + nm(e['target']) + r'} \quad (rank ' + f"{e['text_rank'] + 1:,} by text, {e['graph_rank'] + 1:,} by graph, {e['fused_rank'] + 1:,} combined)" + r'} \\', r'\midrule',
          r'\emph{Text representation} & \emph{Graph representation} & \emph{Combined} \\']
    for r in range(5): L.append(f"{r + 1}. {hit(e['text_top'][r])} & {r + 1}. {hit(e['graph_top'][r])} & {r + 1}. {hit(e['fused_top'][r])} \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\par\vspace{3mm}']
open(PAPER + 'figs/fig_case.tex', 'w').write('\n'.join(L[:-1]) + '\n'); print('\n'.join(L))
