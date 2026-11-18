"""Real vs degree-preserving rewired co-purchase graphs, from the graph files of scripts 01 and 02; no training.
Katcher's: real = <GRAPHS>g1/graph_edges.csv, rewired = <GRAPHS>g1/rewire/adj_rewire_seed{42..46}.npz.
Instacart: real = <IC>graph/adj_freq3.npz, rewired = <IC_GRAPHS>rw_co_{S}/rewire/adj_freq3_rewire_seed{S}.npz.
Per item: (1) unweighted degree equal?  (2) weighted degree s_i = sum_j count_ij.  (3) the values LightGCN uses: LGConv / gcn_norm with
add_self_loops=False and edge weight w = count / max(count): d_i = sum_j w_ij, the factor d_i^(-1/2), and the row sum of the normalised
adjacency r_i = sum_j w_ij / sqrt(d_i d_j). Statistics over items with degree > 0.
-> <PAPER>/tables/rewire_degree.json, <PAPER>/tables/T_rewire_degree.tex"""
import json, os, sys
import numpy as np
import scipy.sparse as sps
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import IC, PAPER, N_ITEMS, SEEDS, GRAPHS as _GRAPHS, IC_GRAPHS as _IC_GRAPHS

GRAPHS = os.path.join(os.path.expanduser(os.environ.get("GRAPHS_DIR", _GRAPHS)), "")
IC_GRAPHS = os.path.join(os.path.expanduser(os.environ.get("IC_GRAPHS_DIR", _IC_GRAPHS)), "")


def sym_from_edges(path, N):
    e = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    i, j, c = e["item_i"].astype(int), e["item_j"].astype(int), e["count"].astype(np.float64)
    return sps.coo_matrix((np.r_[c, c], (np.r_[i, j], np.r_[j, i])), shape=(N, N)).tocsr()


def load_npz(path):
    return sps.load_npz(path).astype(np.float64).tocsr()


def check(A, N):
    assert A.shape == (N, N) and (abs(A - A.T)).nnz == 0 and A.diagonal().sum() == 0 and (A.data > 0).all()
    return A


def quantities(A):
    deg = np.asarray((A > 0).sum(1)).ravel()
    s = np.asarray(A.sum(1)).ravel()                          # weighted degree (pair counts)
    W = A / A.data.max()                                      # LightGCN edge weight count / max(count)
    d = np.asarray(W.sum(1)).ravel()                          # gcn_norm degree
    inv = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
    Wc = W.tocoo(); r = np.zeros(A.shape[0])
    np.add.at(r, Wc.row, Wc.data * inv[Wc.row] * inv[Wc.col])  # row sum of D^-1/2 W D^-1/2
    return {"deg": deg, "weighted_degree": s, "lgcn_degree_d": d, "lgcn_inv_sqrt_d": inv, "lgcn_norm_row_sum": r}, A.data.max()


def compare(real, rw, mask):
    a, b = real[mask], rw[mask]
    rel = np.abs(b - a) / a
    return {"spearman": float(spearmanr(a, b).statistic), "pearson": float(pearsonr(a, b).statistic),
            "rel_change_mean": float(rel.mean()), "rel_change_median": float(np.median(rel)), "rel_change_p90": float(np.percentile(rel, 90)),
            "share_rel_change_gt_10pct": float(np.mean(rel > 0.10)), "share_rel_change_gt_50pct": float(np.mean(rel > 0.50))}


KEYS = ["weighted_degree", "lgcn_degree_d", "lgcn_inv_sqrt_d", "lgcn_norm_row_sum"]
out = {"definition": __doc__, "datasets": {}}
for ds, N, real_path, rw_paths in [
        ("katchers", N_ITEMS["kc"], GRAPHS + "g1/graph_edges.csv", {s: GRAPHS + f"g1/rewire/adj_rewire_seed{s}.npz" for s in SEEDS}),
        ("instacart", N_ITEMS["ic"], IC + "graph/adj_freq3.npz", {s: IC_GRAPHS + f"rw_co_{s}/rewire/adj_freq3_rewire_seed{s}.npz" for s in SEEDS})]:
    R = check(sym_from_edges(real_path, N) if real_path.endswith(".csv") else load_npz(real_path), N)
    qr, mr = quantities(R)
    mask = qr["deg"] > 0
    res = {"real": real_path, "items": N, "items_with_edges": int(mask.sum()), "edges": int(R.nnz // 2), "max_count_real": float(mr), "seeds": {}}
    rw_q = {}
    for s, p in rw_paths.items():
        A = check(load_npz(p), N); q, m = quantities(A); rw_q[s] = q
        extra = {}
        if ds == "instacart":                                 # the copy that LightGCN read
            g = IC_GRAPHS + f"rw_co_{s}/graph/adj_freq3.npz"
            extra["graph_copy_identical"] = bool(os.path.exists(g) and abs(load_npz(g) - A).nnz == 0)
        res["seeds"][s] = {"file": p, "edges": int(A.nnz // 2), "degree_identical_per_item": bool(np.array_equal(q["deg"], qr["deg"])),
                           "weight_multiset_identical": bool(np.array_equal(np.sort(A.data), np.sort(R.data))), "max_count": float(m),
                           "real_edges_kept_share": float(R.multiply(A > 0).nnz / R.nnz), **extra,
                           **{k: compare(qr[k], q[k], mask) for k in KEYS}}
    res["mean_over_seeds"] = {k: {st: float(np.mean([res["seeds"][s][k][st] for s in SEEDS])) for st in res["seeds"][SEEDS[0]][k]} for k in KEYS}
    res["range_over_seeds"] = {k: {st: [float(min(res["seeds"][s][k][st] for s in SEEDS)), float(max(res["seeds"][s][k][st] for s in SEEDS))]
                                   for st in res["seeds"][SEEDS[0]][k]} for k in KEYS}
    res["seed_averaged_rewired_value"] = {k: compare(qr[k], np.mean([rw_q[s][k] for s in SEEDS], 0), mask) for k in KEYS}
    res["all_seeds_degree_identical"] = all(res["seeds"][s]["degree_identical_per_item"] for s in SEEDS)
    out["datasets"][ds] = res

os.makedirs(PAPER + "tables", exist_ok=True)
json.dump(out, open(PAPER + "tables/rewire_degree.json", "w"), indent=1)

NAMES = {"weighted_degree": "Weighted degree $\\sum_j c_{ij}$", "lgcn_inv_sqrt_d": "LightGCN factor $d_i^{-1/2}$",
         "lgcn_norm_row_sum": "Row sum of $D^{-1/2}WD^{-1/2}$"}
tex = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
       " & & Spearman & Pearson & \\multicolumn{3}{c}{$|\\mathrm{rw}-\\mathrm{real}|/\\mathrm{real}$} & $>$10\\% & $>$50\\% \\\\",
       "\\cmidrule(lr){5-7}", "Dataset & Per-item value & & & mean & median & 90th pct. & & \\\\", "\\midrule"]
for ds, lab in (("katchers", "Katcher's"), ("instacart", "Instacart")):
    for k in NAMES:
        v = out["datasets"][ds]["mean_over_seeds"][k]
        tex.append(f"{lab} & {NAMES[k]} & {v['spearman']:.3f} & {v['pearson']:.3f} & {v['rel_change_mean']:.3f} & {v['rel_change_median']:.3f} & "
                   f"{v['rel_change_p90']:.3f} & {100 * v['share_rel_change_gt_10pct']:.1f}\\% & {100 * v['share_rel_change_gt_50pct']:.1f}\\% \\\\")
        lab = ""
tex += ["\\bottomrule", "\\end{tabular}"]
open(PAPER + "tables/T_rewire_degree.tex", "w").write("\n".join(tex) + "\n")
print(json.dumps({ds: {"all_seeds_degree_identical": r["all_seeds_degree_identical"], "mean_over_seeds": r["mean_over_seeds"]} for ds, r in out["datasets"].items()}, indent=1))
