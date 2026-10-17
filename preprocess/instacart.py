"""Preprocess the Instacart dataset used in the paper.

Inputs expected in --raw-dir:
  orders.csv
  order_products__prior.csv
  order_products__train.csv
  products.csv
  aisles.csv
  departments.csv

Outputs in --out-dir:
  split.json
  target.json
  negative_test.json
  item_meta.json
  user_item_history.json
  graph/adj_freq1.npz
  graph/adj_freq3.npz
  graph/adj_freq5.npz

Steps:
department filtering -> active-user filtering -> seeded user sampling ->
iterative 5-core -> user-level 60/20/20 split -> last-transaction target
construction -> fixed 1:49 test negatives -> train-history-only co-purchase graphs.
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz
from sklearn.model_selection import GroupShuffleSplit

# Dataset-selection constants preserved from the cleaned Instacart script.
CORE = 5
KEEP_DEPARTMENTS = ["produce", "dairy eggs"]
MAX_ORDERS = 80
MIN_ORDERS = 10
SEED = 42
TARGET_USERS = 16840


def kcore(df, k=5):
    while True:
        uc = df.groupby("user_id")["product_id"].nunique()
        ic = df.groupby("product_id")["user_id"].nunique()
        bad_u = set(uc[uc < k].index)
        bad_i = set(ic[ic < k].index)
        if not bad_u and not bad_i:
            break
        df = df[
            ~df["user_id"].isin(bad_u)
            & ~df["product_id"].isin(bad_i)
        ]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    graph_dir = out_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    orders = pd.read_csv(raw_dir / "orders.csv")
    prior = pd.read_csv(raw_dir / "order_products__prior.csv")
    train = pd.read_csv(raw_dir / "order_products__train.csv")
    products = pd.read_csv(raw_dir / "products.csv")
    aisles = pd.read_csv(raw_dir / "aisles.csv")
    depts = pd.read_csv(raw_dir / "departments.csv")

    order_products = pd.concat([prior, train], ignore_index=True)
    prod = (
        products
        .merge(aisles, on="aisle_id", how="left")
        .merge(depts, on="department_id", how="left")
    )

    # Product-group filtering.
    keep_pids = set(
        prod[prod["department"].isin(KEEP_DEPARTMENTS)]["product_id"]
    )
    op = order_products[
        order_products["product_id"].isin(keep_pids)
    ].copy()

    # Active users based on the number of retained transactions.
    op = op.merge(
        orders[["order_id", "user_id", "order_number"]],
        on="order_id",
        how="left",
    )
    user_norders = op.groupby("user_id")["order_id"].nunique()
    active = user_norders[
        (user_norders >= MIN_ORDERS)
        & (user_norders <= MAX_ORDERS)
    ].index

    if len(active) > TARGET_USERS:
        sampled = set(
            rng.choice(
                np.array(active),
                size=TARGET_USERS,
                replace=False,
            ).tolist()
        )
    else:
        sampled = set(np.array(active).tolist())

    op = op[op["user_id"].isin(sampled)].copy()

    # Iterative 5-core.
    op = kcore(op, CORE)

    item_ids = sorted(op["product_id"].unique())
    item2idx = {pid: i for i, pid in enumerate(item_ids)}
    num_items = len(item_ids)

    # User-level 60/20/20 split.
    users = np.array(sorted(op["user_id"].unique()))
    gss1 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.6,
        random_state=SEED,
    )
    train_idx, holdout_idx = next(
        gss1.split(users, groups=users)
    )
    train_users = set(users[train_idx].tolist())
    holdout_users_arr = users[holdout_idx]

    gss2 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.5,
        random_state=SEED,
    )
    val_rel, test_rel = next(
        gss2.split(
            holdout_users_arr,
            groups=holdout_users_arr,
        )
    )
    val_users = set(holdout_users_arr[val_rel].tolist())
    test_users = set(holdout_users_arr[test_rel].tolist())

    # Last retained transaction -> target transaction.
    u_maxord = op.groupby("user_id")["order_number"].max()
    op_max = op.merge(
        u_maxord.rename("max_on"),
        on="user_id",
    )
    is_last = op_max["order_number"] == op_max["max_on"]

    target = {}
    for uid, grp in op_max[is_last].groupby("user_id"):
        candidates = grp["product_id"].unique()
        selected_pid = rng.choice(candidates)
        target[int(uid)] = int(item2idx[selected_pid])

    # History excludes the held-out last transaction.
    op_hist = op_max[~is_last].copy()

    # Fixed 1:49 test candidates.
    hist_items_by_user = (
        op_hist.groupby("user_id")["product_id"]
        .apply(
            lambda s: set(
                item2idx[p]
                for p in s.unique()
            )
        )
        .to_dict()
    )

    all_idx = np.arange(num_items)
    negative_test = {}
    for uid in sorted(test_users):
        pos = target[int(uid)]
        seen = hist_items_by_user.get(uid, set()) | {pos}
        pool = np.setdiff1d(
            all_idx,
            np.array(list(seen)),
            assume_unique=False,
        )
        negs = rng.choice(
            pool,
            size=49,
            replace=False,
        ).tolist()
        negative_test[str(uid)] = {
            "pos": int(pos),
            "negs": [int(x) for x in negs],
        }

    # Co-purchase graph: training users' input histories only.
    op_graph = op_hist[
        op_hist["user_id"].isin(train_users)
    ].copy()

    edge_freq = defaultdict(int)
    for _, grp in op_graph.groupby("order_id"):
        items = [
            item2idx[p]
            for p in grp["product_id"].unique()
        ]
        if len(items) < 2:
            continue
        for a, b in combinations(sorted(items), 2):
            edge_freq[(a, b)] += 1

    for cut in [1, 3, 5]:
        rows, cols, vals = [], [], []
        for (a, b), freq in edge_freq.items():
            if freq < cut:
                continue
            rows += [a, b]
            cols += [b, a]
            vals += [freq, freq]

        adj = coo_matrix(
            (vals, (rows, cols)),
            shape=(num_items, num_items),
        ).tocsr()
        save_npz(graph_dir / f"adj_freq{cut}.npz", adj)

        connected = len(set(rows))
        isolated_ratio = (
            (num_items - connected) / num_items
            if num_items else 0.0
        )
        undirected_edges = sum(
            1 for freq in edge_freq.values()
            if freq >= cut
        )
        print(
            f"freq={cut}: edges={undirected_edges:,}, "
            f"isolated={isolated_ratio*100:.1f}%"
        )

    # Item metadata in item_idx order.
    prod_lookup = prod.set_index("product_id")
    item_meta = {}
    for pid, idx in item2idx.items():
        row = prod_lookup.loc[pid]
        item_meta[str(idx)] = {
            "product_id": int(pid),
            "text": (
                f"{row['department']} - "
                f"{row['aisle']} - "
                f"{row['product_name']}"
            ),
        }

    # Ordered per-user transaction histories.
    grouped = (
        op_hist.assign(
            idx=op_hist["product_id"].map(item2idx)
        )
        .groupby(
            ["user_id", "order_number", "order_id"]
        )["idx"]
        .apply(list)
        .reset_index()
    )

    user_item_history = {}
    for uid, ug in grouped.groupby("user_id"):
        ug = ug.sort_values("order_number")
        user_item_history[str(uid)] = [
            [int(x) for x in items]
            for items in ug["idx"]
        ]

    split = {
        "train": sorted(int(x) for x in train_users),
        "val": sorted(int(x) for x in val_users),
        "test": sorted(int(x) for x in test_users),
    }

    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump(split, f)

    with open(out_dir / "target.json", "w", encoding="utf-8") as f:
        json.dump(
            {str(u): int(i) for u, i in target.items()},
            f,
        )

    with open(
        out_dir / "negative_test.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(negative_test, f)

    with open(
        out_dir / "item_meta.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            item_meta,
            f,
            ensure_ascii=False,
        )

    with open(
        out_dir / "user_item_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(user_item_history, f)

    basket = op.groupby("order_id")["product_id"].nunique()
    multi_ratio = float((basket >= 2).mean())

    print(
        f"Final: users={op['user_id'].nunique():,}, "
        f"items={num_items:,}, "
        f"transactions={op['order_id'].nunique():,}, "
        f"avg_basket={basket.mean():.4f}, "
        f"multi_ratio={multi_ratio*100:.2f}%"
    )
    print(
        "split train/val/test:",
        len(train_users),
        len(val_users),
        len(test_users),
    )


if __name__ == "__main__":
    main()
