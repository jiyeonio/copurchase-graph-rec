"""Katcher's co-purchase graphs from the input histories of the training users: two distinct items are linked when they are bought in the
same order, weighted by the number of such orders. f >= 1 is the graph of the paper (g1); f >= 2 and f >= 3 are used in the appendix on
the co-purchase threshold (gf2, gf3). -> <GRAPHS>/{g1,gf2,gf3}/graph_edges.csv, stats.json"""
import csv, itertools, json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import KC, GRAPHS, N_ITEMS
N = N_ITEMS["kc"]
train = json.load(open(KC + "train_data.json"))
assert not (set(train) & (set(json.load(open(KC + "val_history.json"))) | set(json.load(open(KC + "test_history.json"))))), "train users overlap val/test users"
same = Counter()
for evs in train.values():
    for e in evs:
        for a, b in itertools.combinations(sorted({int(i) for i in e["items"] if 0 <= int(i) < N}), 2):
            same[(a, b)] += 1
for name, f in (("g1", 1), ("gf2", 2), ("gf3", 3)):
    counts = {k: v for k, v in same.items() if v >= f}; d = GRAPHS + name
    os.makedirs(d, exist_ok=True); keys = sorted(counts); mx = max(counts.values())
    with open(f"{d}/graph_edges.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["item_i", "item_j", "count", "weight"])
        for a, b in keys: w.writerow([a, b, counts[(a, b)], counts[(a, b)] / mx])
    deg = np.zeros(N, int)
    for a, b in keys: deg[a] += 1; deg[b] += 1
    st = dict(min_count=f, edges=len(keys), isolated_items=int((deg == 0).sum()), isolated_ratio=float((deg == 0).mean()), max_count=int(mx))
    json.dump(st, open(f"{d}/stats.json", "w"), indent=1); print(name, st, flush=True)
