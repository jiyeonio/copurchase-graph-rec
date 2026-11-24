"""Users of the case-study figure (Figure 6): the test users of Instacart that meet the selection conditions, from the dump of the figure
(<AN>/case_ic_seed42.json: --dump-case of model_instacart.py, seed 42 checkpoint of the proposed model; the file read by make_case.py).
Conditions: the target is a new item (not in the history); at most 12 orders; each of the last three orders has at most 6 items; and
  graph side: graph rank <= 10, text rank > 30, combined rank <= 10;  text side: text rank <= 10, graph rank > 30, combined rank <= 10.
Candidates are listed by combined rank. Per candidate: ranks, orders, items, target group of Table 8 (link_an_lg.py), co-purchase degree of
the target, and the orders that hold history items linked to the target by a co-purchase edge (graph of link_an_lg.py: training users,
f >= 3). Also the number of test users meeting the rank conditions alone. CPU only. -> <AN>/case_candidates.json"""
import json
from common import AN, IC, FMIN, data, cograph
from link_an_lg import feats

FIGURE_USERS = {'197342': 'User A', '36963': 'User B'}      # USERS of make_case.py
U = json.load(open(AN + 'case_ic_seed42.json'))['users']; im = json.load(open(IC + 'item_meta.json')); nm = lambda i: im[str(i)]['text'].split(' - ')[-1]


def orders(e):
    o = {}
    for it, t in zip(e['history'], e['tx']): o.setdefault(t, []).append(it)
    return [o[k] for k in sorted(o)]


graph_side = lambda e: e['graph_rank'] < 10 and e['text_rank'] >= 30 and e['fused_rank'] < 10     # ranks in the dump are 0-based
text_side = lambda e: e['text_rank'] < 10 and e['graph_rank'] >= 30 and e['fused_rank'] < 10
A, B, base = [], [], 0
for u, e in U.items():
    if e['target'] in set(e['history']) or len(set(e['tx'])) > 12: continue
    if max(len(o) for o in orders(e)[-3:]) > 6: continue
    base += 1
    if graph_side(e): A.append((e['fused_rank'], u))
    if text_side(e): B.append((e['fused_rank'], u))
F, _ = feats('ic'); _, _, trh = data('ic', max_orders=None); nbr = cograph(trh, FMIN['ic'])
rows = []
for side, X in (('graph', A), ('text', B)):
    for _, u in sorted(X):
        e = U[u]; od = orders(e); t = e['target']; nb = set(nbr[t])
        linked = [dict(order=k - len(od), items=[nm(i) for i in o if i in nb]) for k, o in enumerate(od) if nb & set(o)]
        rows.append(dict(user=u, side=side, in_figure=FIGURE_USERS.get(u, ''), target=nm(t),
                         text_rank=e['text_rank'] + 1, graph_rank=e['graph_rank'] + 1, fused_rank=e['fused_rank'] + 1,
                         orders=len(od), items=len(e['history']), distinct_items=len(set(e['history'])),
                         table8_group=F[u]['rel'], target_degree=F[u]['deg'], linked_history=linked))
out = dict(test_users=len(U), pass_order_conditions=base, graph_side=[u for _, u in sorted(A)], text_side=[u for _, u in sorted(B)],
           rank_conditions_only=dict(graph_side=sum(map(graph_side, U.values())), text_side=sum(map(text_side, U.values()))), rows=rows)
json.dump(out, open(AN + 'case_candidates.json', 'w'), indent=1, ensure_ascii=False)
print(f"test users {len(U):,}; order conditions met {base:,}; graph side {out['graph_side']}; text side {out['text_side']}; "
      f"rank conditions alone: graph side {out['rank_conditions_only']['graph_side']:,}, text side {out['rank_conditions_only']['text_side']:,}")
for r in rows:
    print(f"{r['user']:>7} {r['side']:5} {r['in_figure']:6} ranks text/graph/combined {r['text_rank']:>5,} {r['graph_rank']:>3} {r['fused_rank']:>3} | orders {r['orders']:>2} "
          f"items {r['items']:>2} ({r['distinct_items']}) | {r['table8_group']} degree {r['target_degree']} | orders with a linked item {len(r['linked_history'])}/{r['orders']}")
