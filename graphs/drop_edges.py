"""Katcher's: remove 25 / 50 / 75% of the co-purchase edges at random. The edges are removed in one fixed order, so the graphs are nested.
-> <GRAPHS>/gdrop{25,50,75}/graph_edges.csv, stats.json"""
import csv, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import GRAPHS, N_ITEMS
DROP_SEED = 2026
rows = list(csv.DictReader(open(GRAPHS + "g1/graph_edges.csv"))); n = len(rows)
perm = np.random.default_rng(DROP_SEED).permutation(n)
for drop in (25, 50, 75):
    keep = sorted(perm[int(round(n * drop / 100)):]); d = f"{GRAPHS}gdrop{drop}"; os.makedirs(d, exist_ok=True)
    assert not os.path.exists(f"{d}/graph_edges.csv")
    with open(f"{d}/graph_edges.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item_i", "item_j", "count", "weight"])
        for k in keep: r = rows[k]; w.writerow([r["item_i"], r["item_j"], r["count"], r["weight"]])
    deg = np.zeros(N_ITEMS["kc"], int)
    for k in keep: deg[int(rows[k]["item_i"])] += 1; deg[int(rows[k]["item_j"])] += 1
    st = dict(edges=len(keep), of=n, isolated_items=int((deg == 0).sum())); json.dump(st, open(f"{d}/stats.json", "w")); print(d, st)
