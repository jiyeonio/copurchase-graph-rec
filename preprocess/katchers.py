"""Prepare Katcher's user histories/targets, the train-only co-purchase graph,
and fixed 1:49 candidate negatives.

Expected input files in --data-dir:
  train_data.csv
  val_data.csv
  test_data.csv
  negative_test.json   # (optional) an existing candidate file to compare with

Outputs written to --output-dir:
  train_data.json / train_targets.json
  val_history.json / val_data.json
  test_history.json / test_data.json
  cold_start_users.json, item2idx.json, user2idx.json
  item_metadata.csv
  co_purchase_edges_train.csv
  negative_test.json   # fixed 1:49 test negatives (target + items of the most recent
                       # 50 orders excluded, np.random.seed(42))

Negative file format (same as the original negative_test.json):
  {"<user_idx>": [item_idx x 49], ...}
"""

import argparse
import json
from itertools import combinations
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

COPURCHASE_MIN_FREQ = 1   # paper setting f >= 1; the graph the model uses is built by graphs/build_graph_katchers.py
SEED = 42
N_NEG = 49


def build_split_dicts(df_split: pd.DataFrame):
    history = {}
    targets = {}
    cold = set()

    for uid, udf in df_split.groupby("user_idx"):
        udf = udf.sort_values("initial_paid_at_dt")
        last_t = udf["initial_paid_at_dt"].max()

        target_rows = udf[udf["initial_paid_at_dt"] == last_t]
        if len(target_rows) != 1:
            raise ValueError(
                f"user {uid}: target rows={len(target_rows)}; expected exactly 1 "
                "for the Katcher's preprocessing used in the experiments"
            )

        target_item = int(target_rows["item_idx"].iloc[0])
        target_day = str(target_rows["initial_paid_at_dt"].iloc[0])

        hist_df = udf[udf["initial_paid_at_dt"] != last_t]
        events = []
        for oid, odf in hist_df.groupby("order_id"):
            events.append(
                {
                    "order_id": str(oid),
                    "items": [int(x) for x in odf["item_idx"].tolist()],
                    "day": str(odf["initial_paid_at_dt"].iloc[0]),
                }
            )
        events.sort(key=lambda x: x["day"])

        uid_str = str(int(uid))
        history[uid_str] = events
        targets[uid_str] = {"items": [target_item], "day": target_day}

        if len(events) <= 3:
            cold.add(int(uid))

    return history, targets, cold


def save_json(obj, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


# ----------------------------------------------------------------------
# Fixed 1:49 test negatives
# ----------------------------------------------------------------------
def get_user_items(events):
    """items of the user's most recent 50 orders"""
    events = events[-50:]
    items = []
    for event in events:
        items.extend(event["items"])
    return list(set(items))


def make_test_negatives(test_history, test_data, num_items, seed=SEED, n_neg=N_NEG):
    """The fixed 1:49 test candidates of the experiments: for every test user, in the order of test_data, n_neg distinct items drawn
    with np.random.choice from all items except the target and the items of the user's most recent 50 orders (global seed 42)."""
    np.random.seed(seed)
    negative_test = {}
    for uid_str, target_event in test_data.items():
        if uid_str not in test_history:
            continue
        target = target_event["items"][0]
        items = get_user_items(test_history[uid_str])
        all_items = set(range(num_items))
        seen_items = set(items) | {target}
        candidate_pool = list(all_items - seen_items)
        if len(candidate_pool) < n_neg:
            continue
        negative_test[uid_str] = np.random.choice(candidate_pool, size=n_neg, replace=False).tolist()
    return negative_test


def check_negatives(name, negs, history, targets, num_items):
    n = t_in = h_users = rep = dup = oob = missing = 0
    for uid, tv in targets.items():
        if uid not in negs:
            missing += 1
            continue
        n += 1
        ns = negs[uid]
        s = set(ns)
        pos = tv["items"][0]
        hs = {i for e in history[uid] for i in e["items"]}
        dup += len(ns) != len(s) or len(ns) != N_NEG
        oob += any((x < 0) or (x >= num_items) for x in ns)
        t_in += pos in s
        h_users += len(s & hs) > 0
        rep += pos in hs
    print(
        f"[{name}] users={n} missing={missing} bad_size_or_dup={dup} out_of_range={oob} | "
        f"target_in_negs={t_in} | users_with_hist_neg={h_users} | "
        f"repeat_target_ratio={rep / max(n, 1):.1%}"
    )
    ok = (missing == 0 and dup == 0 and oob == 0 and t_in == 0 and h_users == 0)
    if not ok:
        raise RuntimeError(f"{name}: negative set violates the history+target exclusion rule")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--min-freq", type=int, default=COPURCHASE_MIN_FREQ)
    args = parser.parse_args()

    np.random.seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(args.data_dir / "train_data.csv")
    df_val = pd.read_csv(args.data_dir / "val_data.csv")
    df_test = pd.read_csv(args.data_dir / "test_data.csv")

    required = {
        "user_id", "order_id", "product_id", "initial_paid_at",
        "product_name", "root_category_name", "category_name", "attribute_values",
    }
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} CSV missing columns: {sorted(missing)}")

    for df in (df_train, df_val, df_test):
        df.dropna(subset=["user_id", "order_id", "product_id", "initial_paid_at"], inplace=True)
        df["user_id"] = df["user_id"].astype(int)
        df["product_id"] = df["product_id"].astype(int)
        df["initial_paid_at_dt"] = pd.to_datetime(df["initial_paid_at"], unit="ms")

    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    unique_items = sorted(df_all["product_id"].unique())
    unique_users = sorted(df_all["user_id"].unique())
    item2idx = {int(item): idx for idx, item in enumerate(unique_items)}
    user2idx = {int(user): idx for idx, user in enumerate(unique_users)}
    num_items = len(unique_items)

    for df in (df_all, df_train, df_val, df_test):
        df["item_idx"] = df["product_id"].map(item2idx)
        df["user_idx"] = df["user_id"].map(user2idx)

    train_data, train_targets, cold_train = build_split_dicts(df_train)
    val_history, val_data, cold_val = build_split_dicts(df_val)
    test_history, test_data, cold_test = build_split_dicts(df_test)
    cold_start_users = cold_train | cold_val | cold_test

    save_json(train_data, args.output_dir / "train_data.json")
    save_json(train_targets, args.output_dir / "train_targets.json")
    save_json(val_history, args.output_dir / "val_history.json")
    save_json(val_data, args.output_dir / "val_data.json")
    save_json(test_history, args.output_dir / "test_history.json")
    save_json(test_data, args.output_dir / "test_data.json")
    save_json(sorted(cold_start_users), args.output_dir / "cold_start_users.json")
    save_json({str(k): int(v) for k, v in item2idx.items()}, args.output_dir / "item2idx.json")
    save_json({str(k): int(v) for k, v in user2idx.items()}, args.output_dir / "user2idx.json")

    item_info = (
        df_all.groupby("item_idx")
        .agg(
            product_name=("product_name", "first"),
            root_category_name=("root_category_name", "first"),
            category_name=("category_name", "first"),
            attribute_values=("attribute_values", "first"),
        )
        .reset_index()
        .sort_values("item_idx")
        .reset_index(drop=True)
    )
    if item_info["item_idx"].tolist() != list(range(num_items)):
        raise RuntimeError("item_idx is not contiguous from 0 to NUM_ITEMS-1")
    item_info.to_csv(args.output_dir / "item_metadata.csv", index=False)

    # Train-only graph, excluding each training user's held-out target row.
    last_t = df_train.groupby("user_idx")["initial_paid_at_dt"].transform("max")
    df_train_graph = df_train[df_train["initial_paid_at_dt"] != last_t].copy()
    # f_ij = number of transactions that contain both items (the co-purchase count of the paper): the DISTINCT items of an order are
    # combined, so a product that appears in several rows of one order counts once and no self-loop arises.
    # (This is the rule of the graphs the experiments used: train_lightgcn_multiseed_katchers_from_trainjson.py.)
    order_items = df_train_graph.groupby("order_id")["item_idx"].apply(
        lambda s: sorted({int(x) for x in s})
    )
    order_items = order_items[order_items.apply(len) >= 2]

    counts = defaultdict(int)
    for items in order_items:
        for i, j in combinations(items, 2):
            counts[(i, j)] += 1

    n_self = sum(1 for (i, j) in counts if i == j)
    if n_self:
        print(f"WARNING: {n_self} self-loop pairs (same item twice in one order) in graph counts")

    filtered = {pair: cnt for pair, cnt in counts.items() if cnt >= args.min_freq}
    if not filtered:
        raise RuntimeError("No co-purchase edges remain after frequency filtering")

    max_count = max(filtered.values())
    edges = [(i, j, cnt / max_count, cnt) for (i, j), cnt in filtered.items()]
    edge_df = pd.DataFrame(edges, columns=["item_i", "item_j", "weight", "frequency"])
    edge_df.to_csv(args.output_dir / "co_purchase_edges_train.csv", index=False)

    # Fixed 1:49 test negatives; a negative_test.json in --data-dir is only compared, never overwritten
    negative_test = make_test_negatives(test_history, test_data, num_items)
    check_negatives("test", negative_test, test_history, test_data, num_items)
    save_json(negative_test, args.output_dir / "negative_test.json")
    neg_path = args.data_dir / "negative_test.json"
    if neg_path.exists():
        with neg_path.open("r", encoding="utf-8") as f:
            given = json.load(f)
        same = given.keys() == negative_test.keys() and all([int(x) for x in given[u]] == negative_test[u] for u in given)
        print(f"negative_test.json identical to {neg_path}: {same}")

    print(f"NUM_USERS={len(unique_users)} NUM_ITEMS={num_items}")
    print(f"split users: train={len(train_data)} val={len(val_data)} test={len(test_data)}")
    print(f"co-purchase edges (freq>={args.min_freq}): {len(edge_df)}")
    print(f"saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()