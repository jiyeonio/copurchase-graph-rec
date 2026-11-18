"""Statistics of the two datasets (the dataset statistics table and the statistics stated in the text) from the raw data. -> <PAPER>/tables/check_stats.json"""
import json, collections, itertools, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import *
out = {}
K = KC
def graph(orders, fmin, N):
    cnt = collections.Counter()
    for o in orders:
        for a, b in itertools.combinations(sorted(set(o)), 2): cnt[(a, b)] += 1
    deg = np.zeros(N, int); E = 0
    for (a, b), c in cnt.items():
        if c >= fmin: deg[a] += 1; deg[b] += 1; E += 1
    return E, deg
def ostats(hists):
    npu = [len(h) for h in hists]; ipo = [len(o) for h in hists for o in h]; ipo_d = [len(set(o)) for h in hists for o in h]
    return dict(orders_per_user_mean=float(np.mean(npu)), orders_per_user_median=float(np.median(npu)), items_per_order_mean=float(np.mean(ipo)), items_per_order_median=float(np.median(ipo)),
                items_per_order_distinct_mean=float(np.mean(ipo_d)), single_item_order_share=float(np.mean(np.array(ipo_d) == 1)), multi_item_order_share=float(np.mean(np.array(ipo_d) > 1)))
tr = json.load(open(K + 'train_data.json'))
for nm in ('train', 'val', 'valid', 'test'):
    for suf in ('_history.json', '_data.json'):
        if os.path.exists(K + nm + suf): d = json.load(open(K + nm + suf)); out[f'kc_{nm}{suf[:-5]}_users'] = len(d)
hk = [[[int(i) for i in o['items']] for o in tr[u]] for u in tr]
out['kc_train_data'] = ostats(hk)
if os.path.exists(K + 'train_history.json'):
    th = json.load(open(K + 'train_history.json')); out['kc_train_history'] = ostats([[[int(i) for i in o['items']] for o in th[u]] for u in th])
for f in (1, 2, 3):
    E, deg = graph([o for h in hk for o in h], f, N_ITEMS["kc"]); out[f'kc_graph_f{f}'] = dict(edges=E, mean_deg=float(deg.mean()), median_deg=float(np.median(deg)), mean_deg_nonisolated=float(deg[deg > 0].mean()), median_deg_nonisolated=float(np.median(deg[deg > 0])), isolated=float((deg == 0).mean()))
I = IC
sp = json.load(open(I + 'split.json')); h = json.load(open(I + 'user_item_history.json')); tg = json.load(open(I + 'target.json'))
out['ic_users'] = {k: len(v) for k, v in sp.items()}; out['ic_items_max_id'] = int(max(max(i for o in hh for i in o) for hh in h.values()))
hi = [h[str(u)] for u in sp['train']]; out['ic_train'] = ostats(hi)
for f in (1, 3):
    E, deg = graph([o for hh in hi for o in hh], f, N_ITEMS["ic"]); out[f'ic_graph_f{f}'] = dict(edges=E, mean_deg=float(deg.mean()), median_deg=float(np.median(deg)), mean_deg_nonisolated=float(deg[deg > 0].mean()), median_deg_nonisolated=float(np.median(deg[deg > 0])), isolated=float((deg == 0).mean()))
# ---------------- distributions (key r1)
# Inputs as above (no raw data), except: rewired graphs (GRAPHS_DIR, IC_GRAPHS_DIR) and, for the Katcher's last-order statistic only,
# the raw log KATCHERS_RAW read as preprocess/katchers_split.py reads it.
from scipy.stats import spearmanr
import scipy.sparse as sps
GR_DIR = os.path.join(os.path.expanduser(os.environ.get("GRAPHS_DIR", GRAPHS)), "")
ICG_DIR = os.path.join(os.path.expanduser(os.environ.get("IC_GRAPHS_DIR", IC_GRAPHS)), "")
def bins(v, B):
    v = np.asarray(v); return {(str(lo) if lo == hi else (f"{lo}+" if hi is None else f"{lo}-{hi}")): float(np.mean((v >= lo) & (v <= (hi if hi is not None else v.max())))) for lo, hi in B}
SIZE_B = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, None)]
LEN_B = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, None)]
W_B = [(1, 1), (2, 2), (3, 5), (6, 10), (11, None)]
def order_size(hists):
    s = [len(set(o)) for h in hists for o in h]; return dict(definition='distinct items per order, input histories of the training users', orders=len(s), shares=bins(s, SIZE_B), max=int(max(s)))
def item_freq(hists, N):
    f = np.zeros(N, int)
    for h in hists:
        for o in h: f[sorted(set(o))] += 1
    return dict(definition='number of orders (input histories of the training users) that contain the item, over all N items', items=N, items_with_0=int((f == 0).sum()),
                **{f'p{q}': float(np.percentile(f, q)) for q in (10, 25, 50, 75, 90, 99)}, mean=float(f.mean()), max=int(f.max()))
def hist_len(hists):
    l = [len(h) for h in hists]; return dict(users=len(l), shares=bins(l, LEN_B))
def pair_cnt(orders):
    cnt = collections.Counter()
    for o in orders:
        for a, b in itertools.combinations(sorted(set(o)), 2): cnt[(a, b)] += 1
    return cnt
def edge_w(cnt, fmin, N):
    c = np.array([v for v in cnt.values() if v >= fmin]); E = len(c)
    return dict(fmin=fmin, edges=E, shares=bins(c, W_B), median=float(np.median(c)), mean=float(c.mean()), max=int(c.max()), density=2 * E / (N * (N - 1)))
def sym(cnt, fmin, N):
    k = [(a, b, v) for (a, b), v in cnt.items() if v >= fmin]; i, j, v = (np.array(x) for x in zip(*k))
    return sps.coo_matrix((np.r_[v, v].astype(float), (np.r_[i, j], np.r_[j, i])), shape=(N, N)).tocsr()
def rewired_spearman(R, files):
    """Spearman between real and rewired weighted degree (sum of pair counts), items with degree > 0 in the real graph"""
    s0 = np.asarray(R.sum(1)).ravel(); m = np.asarray((R > 0).sum(1)).ravel() > 0; res = {}
    for s, p in files.items():
        Wr = sps.load_npz(p).astype(float).tocsr(); res[str(s)] = float(spearmanr(s0[m], np.asarray(Wr.sum(1)).ravel()[m]).statistic)
    return res
R1 = {}
kc_hist = {'train': hk, 'val': [[[int(i) for i in o['items']] for o in v] for v in json.load(open(K + 'val_history.json')).values()],
           'test': [[[int(i) for i in o['items']] for o in v] for v in json.load(open(K + 'test_history.json')).values()]}
ic_hist = {k: [[[int(i) for i in o] for o in h[str(u)]] for u in sp[k]] for k in ('train', 'val', 'test')}
for ds, H_, N, fmin in (('kc', kc_hist, N_ITEMS['kc'], FMIN['kc']), ('ic', ic_hist, N_ITEMS['ic'], FMIN['ic'])):
    cnt = pair_cnt([o for hh in H_['train'] for o in hh])
    R = sym(cnt, fmin, N)
    if ds == 'kc':
        ge = np.genfromtxt(GR_DIR + 'g1/graph_edges.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
        F_ = sps.coo_matrix((np.r_[ge['count'], ge['count']].astype(float), (np.r_[ge['item_i'], ge['item_j']], np.r_[ge['item_j'], ge['item_i']])), shape=(N, N)).tocsr()
        real_file, rw = GR_DIR + 'g1/graph_edges.csv', {s: GR_DIR + f'g1/rewire/adj_rewire_seed{s}.npz' for s in SEEDS}
    else:
        F_ = sps.load_npz(I + 'graph/adj_freq3.npz').astype(float).tocsr()
        real_file, rw = I + 'graph/adj_freq3.npz', {s: ICG_DIR + f'rw_co_{s}/rewire/adj_freq3_rewire_seed{s}.npz' for s in SEEDS}
    R1[ds] = dict(order_size_train=order_size(H_['train']), item_purchase_frequency_train=item_freq(H_['train'], N),
                  history_length_orders={k: hist_len(v) for k, v in H_.items()},
                  edge_weight=dict(**edge_w(cnt, fmin, N), rebuilt_identical_to_graph_file=bool(abs(R - F_).nnz == 0), graph_file=real_file),
                  rewired_weighted_degree_spearman=dict(per_seed=rewired_spearman(R, rw), files=rw))
# consistency with tables/rewire_degree_check.py (rewire_degree.json, if given)
RWJ = os.environ.get('REWIRE_DEGREE_JSON', PAPER + 'tables/rewire_degree.json')
if os.path.exists(RWJ):
    J = json.load(open(RWJ))
    for ds, nm in (('kc', 'katchers'), ('ic', 'instacart')):
        ref = {s: J['datasets'][nm]['seeds'][s]['weighted_degree']['spearman'] for s in J['datasets'][nm]['seeds']}
        R1[ds]['rewired_weighted_degree_spearman']['equals_rewire_degree_json'] = bool(all(abs(ref[s] - v) < 1e-12 for s, v in R1[ds]['rewired_weighted_degree_spearman']['per_seed'].items()))
        R1[ds]['rewired_weighted_degree_spearman']['rewire_degree_json'] = RWJ
# single-item share of the last order
RAW = os.environ.get('KATCHERS_RAW')
if RAW and os.path.exists(RAW):
    import pandas as pd
    data = pd.read_csv(RAW, sep='\t', encoding='utf-8', low_memory=False)   # read as in preprocess/katchers_split.py
    data['unique_purchase_dates'] = data.groupby('user_id')['initial_paid_at'].transform(lambda x: x.nunique())
    data = data[data['unique_purchase_dates'] > 1].drop(columns=['unique_purchase_dates'])
    df_grouped = data.groupby('user_id')['initial_paid_at'].max().reset_index(name='initial_paid_at')
    df_max_all = data.merge(df_grouped, how='inner', on=['user_id', 'initial_paid_at'])
    KC_TEST_CSV = os.environ.get('KATCHERS_TEST_CSV', KC + 'test_data.csv')   # split file of the test users
    tst = pd.read_csv(KC_TEST_CSV)
    tu = set(tst.dropna(subset=['user_id', 'order_id', 'product_id', 'initial_paid_at'])['user_id'].astype(int))
    g = df_max_all[pd.to_numeric(df_max_all['user_id'], errors='coerce').isin(tu)].groupby('user_id')
    nd = g['product_id'].nunique()
    R1['q8_kc_last_order_single_item'] = dict(definition='test users; rows of the last initial_paid_at before the target draw (df_max_all of preprocess/katchers_split.py); share with one distinct product_id',
                                              raw=RAW, read='pd.read_csv(sep="\\t"), as preprocess/katchers_split.py', test_csv=KC_TEST_CSV, test_users=len(tu), users_found=int(len(nd)),
                                              single_item_share=float((nd == 1).mean()), rows_per_user_mean=float(g.size().mean()),
                                              users_with_more_than_one_order_id_at_last_time=int((g['order_id'].nunique() > 1).sum()),
                                              distinct_items_counts={str(k): int(v) for k, v in nd.value_counts().sort_index().items()})
else:
    R1['q8_kc_last_order_single_item'] = dict(available=False, reason='KATCHERS_RAW not given')
R1['q8_ic_last_order_single_item'] = dict(available=False, reason='needs op_max[is_last] of preprocess/instacart.py, i.e. the Instacart raw files (orders.csv, order_products__*.csv), which are not an input of this script. target.json keeps only the drawn target item, not the whole last order.')
out['r1'] = R1
json.dump(out, open(PAPER + 'tables/check_stats.json', 'w'), indent=1); print(json.dumps(out, indent=1))
