"""The two fixed item representations (text embedding; LightGCN embedding of the co-purchase graph, seed 42):
(i) nearest-neighbour statistics of both datasets -> <PAPER>/tables/emb_knn.json, (ii) t-SNE map of the Instacart items -> <PAPER>/figs/emb_tsne_ic.pdf"""
import json, collections, itertools, numpy as np
from common import *
KNN_K, TSNE_SEED, TOP_GROUPS = 10, 0, 6
OUT = PAPER
def kc():
    meta = json.load(open(AN + 'kc_item_meta.json'))
    text = np.load(KC_TEXT); graph = np.load(GRAPHS + 'g1/emb/item_embeddings_graph_seed42.npy'); N = len(text)
    EN = {'식품': 'Food', '뷰티': 'Beauty', '생활용품': 'Household', '주방용품': 'Kitchen', '패션의류잡화': 'Fashion', '출산/유아동': 'Baby & kids', '가구/홈데코': 'Home decor', '자동차용품': 'Car'}
    cats = [EN.get(meta[str(i)]['root_category'], meta[str(i)]['root_category']) if str(i) in meta else '?' for i in range(N)]
    fine = [meta[str(i)]['category'] if str(i) in meta else '?' for i in range(N)]
    tr = json.load(open(KC + 'train_data.json')); orders = [sorted(set(int(i) for i in o['items'])) for u in tr for o in tr[u]]
    return "Katcher's", text, graph, cats, fine, orders, FMIN['kc']
def ic():
    im = json.load(open(IC + 'item_meta.json')); N = len(im)
    text = np.load(IC_TEXT); graph = np.load(IC + 'item_embeddings_graph_freq3_seed42.npy')
    aisle = [im[str(i)]['text'].split(' - ')[1] for i in range(N)]
    sp = json.load(open(IC + 'split.json')); h = json.load(open(IC + 'user_item_history.json')); orders = [sorted(set(o)) for u in sp['train'] for o in h[str(u)]]
    return 'Instacart', text, graph, aisle, aisle, orders, FMIN['ic']
def edges(orders, fmin, N):
    cnt = collections.Counter()
    for o in orders:
        for a, b in itertools.combinations(o, 2): cnt[(a, b)] += 1
    A = np.zeros((N, N), bool)
    for (a, b), c in cnt.items():
        if c >= fmin: A[a, b] = A[b, a] = True
    return A
def knn(X, k=KNN_K):
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12); S = Xn @ Xn.T; np.fill_diagonal(S, -np.inf); return np.argsort(-S, axis=1)[:, :k]
RES = {}; DATA = []
for load in (kc, ic):
    name, text, graph, cats, fine, orders, fmin = load(); N = len(text); A = edges(orders, fmin, N); deg = A.sum(1); cat_arr = np.array(cats); fine_arr = np.array(fine)
    same = cat_arr[:, None] == cat_arr[None, :]; np.fill_diagonal(same, False); samef = fine_arr[:, None] == fine_arr[None, :]; np.fill_diagonal(samef, False)
    conn = deg > 0
    R = {'n_items': N, 'isolated': float((~conn).mean()), 'n_categories': len(set(cats)), 'n_fine': len(set(fine)),
         'random': {'cat': float(same.sum(1).mean() / (N - 1)), 'fine': float(samef.sum(1).mean() / (N - 1)), 'link': float(deg.mean() / (N - 1)), 'link_conn': float(deg[conn].mean() / (N - 1))}}
    for nm, X in (('text', text), ('graph', graph)):
        nb = knn(X); r = np.arange(N)[:, None]
        R[nm] = {'cat': float(same[r, nb].mean()), 'fine': float(samef[r, nb].mean()), 'link': float(A[r, nb].mean()), 'link_conn': float(A[r, nb][conn].mean()), 'cat_conn': float(same[r, nb][conn].mean())}
    RES[name] = R; DATA.append((name, text, graph, cats, deg))
    print(name, json.dumps(R, indent=0).replace('\n', ' '))
json.dump(RES, open(OUT + 'tables/emb_knn.json', 'w'), indent=1)
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
def tsne2(X):
    Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); P = Xc @ Vt[:min(50, X.shape[1])].T
    return TSNE(n_components=2, perplexity=30, init='pca', random_state=TSNE_SEED, metric='cosine').fit_transform(P)
PAL = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#17becf']
# ---- t-SNE map of the Instacart items in the two representations
name, text, graph, cats, deg = DATA[1]; top = [c for c, _ in collections.Counter(cats).most_common(TOP_GROUPS)]; cat_arr = np.array(cats); iso = deg == 0
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
for ax, (X, rep) in zip(axes, ((text, 'Text representation'), (graph, 'Graph representation'))):
    Z = tsne2(X); other = ~np.isin(cat_arr, top)
    ax.scatter(Z[other & ~iso, 0], Z[other & ~iso, 1], s=5, color='0.75', linewidths=0, alpha=0.7, label='other aisles', rasterized=True)
    ax.scatter(Z[other & iso, 0], Z[other & iso, 1], s=9, facecolors='none', edgecolors='0.6', linewidths=0.5, rasterized=True)
    for ci, cname in enumerate(top):
        m = cat_arr == cname
        ax.scatter(Z[m & ~iso, 0], Z[m & ~iso, 1], s=7, color=PAL[ci], linewidths=0, alpha=0.8, label=cname, rasterized=True)
        ax.scatter(Z[m & iso, 0], Z[m & iso, 1], s=11, facecolors='none', edgecolors=PAL[ci], linewidths=0.6, rasterized=True)
    ax.set_title(rep, fontsize=10.5); ax.set_xticks([]); ax.set_yticks([])
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, fontsize=9, frameon=False, markerscale=2.4, handletextpad=0.2, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(rect=(0, 0.14, 1, 1)); plt.savefig(OUT + 'figs/emb_tsne_ic.pdf', dpi=300); plt.close(); print('saved')
