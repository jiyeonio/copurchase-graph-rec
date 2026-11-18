"""The data of the paper, read as it is, and the purchase records that carry the text. Nothing in either folder is written or changed.

Two inputs:
  data folder   paths.KC         output of preprocess/katchers.py. It defines who is a training / validation / test user, the history
                                 as lists of item indices, the held-out target and the fixed 1:49 test candidates. It is read with the
                                 rules of the baselines (baselines/original/kc_sasrec.py); check_protocol.py compares the two.
  split folder  paths.KC_SPLIT   train_data.csv, val_data.csv, test_data.csv, the input of preprocess/katchers.py: one row per purchased
                                 product with its text (category_name, product_name, attribute_values) and its payment time.

BERT-NSP reads sentences, so it needs the rows. The split, the targets and the candidates still come from the data folder; none of them
is made again. match() checks that both inputs describe the same users, orders and targets, and stops otherwise.

  history      the last 50 orders of a user in time order (no Katcher's user has more than 50)
  target       the single held-out item of the user
  validation   target against 49 negatives drawn per user with a fixed seed from all items except the target (rule of the baselines)
  test         target against the 49 negatives of negative_test.json
  seen         the items of the whole history: removed in the masked full-catalogue ranking and used for the novel / repeat groups
"""
import hashlib
import json
import os
import zlib

import numpy as np
import pandas as pd

MAX_ORDERS = 50            # history window of the paper
N_NEG = 49
VAL_NEGATIVE_SEED = 42
SPLITS = ("train", "val", "test")
ID_COLUMNS = ("user_id", "order_id", "product_id", "initial_paid_at")
TEXT_COLUMNS = ("category_name", "product_name", "attribute_values")


class Split:
    """users: evaluated users; orders[u]: lists of item indices; target[u]; cand[u] = [target, 49 negatives]; seen[u]: set of items"""

    def __init__(self):
        self.users, self.orders, self.target, self.cand, self.seen = [], {}, {}, {}, {}

    def add(self, uid, orders, target, negatives, seen):
        self.users.append(uid)
        self.orders[uid], self.target[uid], self.cand[uid], self.seen[uid] = orders, target, [target, *negatives], seen


class Dataset:
    """num_items, train (Split without candidates), val and test (Split), sha256 (of every file read)"""


def _read(folder, name, sha256):
    with open(os.path.join(folder, name), "rb") as f:                 # opened for reading only
        raw = f.read()
    sha256[name] = hashlib.sha256(raw).hexdigest()
    return raw


def _target(value, n, uid):
    if isinstance(value, dict):
        value = value.get("items", [])
    if isinstance(value, list):
        if not value:
            raise ValueError(f"user {uid}: empty target")
        value = value[0]
    t = int(value)
    if not 0 <= t < n:
        raise ValueError(f"user {uid}: target {t} outside the catalogue of {n} items")
    return t


def _fixed_negatives(value, target, n, uid):
    negs = [int(x) for x in (value.get("negs", []) if isinstance(value, dict) else value)][:N_NEG]
    if len(negs) != N_NEG or len(set(negs)) != N_NEG or target in negs or any(not 0 <= x < n for x in negs):
        raise ValueError(f"user {uid}: negative_test.json does not hold 49 distinct negatives without the target")
    return negs


def load(folder, num_items=None):
    """the data folder, read with the rules of the baselines (baselines/original/kc_sasrec.py)"""
    sha256 = {}
    js = {k: json.loads(_read(folder, k + ".json", sha256)) for k in
          ("train_data", "train_targets", "val_history", "val_data", "test_history", "test_data", "negative_test")}
    for k in ("item2idx", "user2idx"):                    # written by preprocess/katchers.py; compared with the split files if present
        js[k] = json.loads(_read(folder, k + ".json", sha256)) if os.path.exists(os.path.join(folder, k + ".json")) else None
    if not num_items and js["item2idx"] is None:
        raise ValueError(f"{folder} has no item2idx.json: give the number of items with --num-items")
    n = int(num_items) if num_items else len(js["item2idx"])

    def orders(events):
        events = sorted(events, key=lambda e: (e.get("day", 0), str(e.get("order_id", ""))))[-MAX_ORDERS:]
        out = [sorted({int(i) for i in e.get("items", []) if 0 <= int(i) < n}) for e in events]
        return [o for o in out if o]

    def seen(events):
        return {int(i) for e in events for i in e.get("items", []) if 0 <= int(i) < n}

    def val_negatives(uid, target):
        pool = np.array([i for i in range(n) if i != target], dtype=np.int64)
        rng = np.random.default_rng((VAL_NEGATIVE_SEED + zlib.crc32(uid.encode("utf-8"))) % (2**32))
        return rng.choice(pool, size=N_NEG, replace=False).tolist()

    ds = Dataset()
    ds.num_items, ds.train, ds.val, ds.test = n, Split(), Split(), Split()
    ds.item2idx, ds.user2idx = ({int(k): int(v) for k, v in js[name].items()} if js[name] is not None else None
                                for name in ("item2idx", "user2idx"))
    for uid, events in js["train_data"].items():
        uid, o = str(uid), orders(events)
        if o:
            ds.train.add(uid, o, _target(js["train_targets"][uid], n, uid), [], seen(events))
    for uid, events in js["val_history"].items():
        uid, o = str(uid), orders(events)
        if o:
            t = _target(js["val_data"][uid], n, uid)
            ds.val.add(uid, o, t, val_negatives(uid, t), seen(events))
    for uid, events in js["test_history"].items():
        uid, o = str(uid), orders(events)
        if o:
            t = _target(js["test_data"][uid], n, uid)
            ds.test.add(uid, o, t, _fixed_negatives(js["negative_test"][uid], t, n, uid), seen(events))
    tr, va, te = set(ds.train.users), set(ds.val.users), set(ds.test.users)
    if tr & va or tr & te or va & te:
        raise ValueError("a user appears in two of train / validation / test")
    if not (tr and va and te):
        raise ValueError("an empty split")
    ds.sha256 = sha256
    return ds


# ---------------------------------------------------------------- the purchase records
class UserRows:
    """One user of a split file. orders: [(payment time, [(item index, category, name, attributes or None), ...]), ...] in time order,
    every purchased row kept (a product bought with two options is two rows). target: the row at the last payment time."""
    __slots__ = ("uid", "orders", "target")

    def __init__(self, uid, orders, target):
        self.uid, self.orders, self.target = uid, orders, target


class Rows:
    """frames[split]: the records of a split file. users[split]: {user index (str): UserRows}, in the order of the file."""


def load_rows(folder):
    """The three split files, read as preprocess/katchers.py reads them: rows without user, order, product or payment time are dropped,
    and items and users are numbered in sorted order over the union of the three files."""
    rows, sha256, frames = Rows(), {}, {}
    for split in SPLITS:
        path = os.path.join(folder, f"{split}_data.csv")
        _read(folder, f"{split}_data.csv", sha256)
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        missing = [c for c in (*ID_COLUMNS, *TEXT_COLUMNS) if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: columns {missing} are missing")
        df = df.dropna(subset=list(ID_COLUMNS)).reset_index(drop=True)
        df["user_id"] = df["user_id"].astype(float).astype("int64")
        df["product_id"] = df["product_id"].astype(float).astype("int64")
        df["order_id"] = df["order_id"].astype(str)
        df["initial_paid_at"] = df["initial_paid_at"].astype("int64")
        frames[split] = df
    union = pd.concat(frames.values(), ignore_index=True)
    rows.frames, rows.sha256 = frames, sha256
    rows.item2idx = {int(p): i for i, p in enumerate(sorted(union["product_id"].unique().tolist()))}
    rows.user2idx = {int(u): i for i, u in enumerate(sorted(union["user_id"].unique().tolist()))}
    rows.users = {split: _user_rows(frames[split], rows) for split in SPLITS}
    return rows


def _user_rows(df, rows):
    codes, uniques = pd.factorize(df["user_id"])                       # users in the order of their first row
    by_user = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
    paid = df["initial_paid_at"].to_numpy()
    item = df["product_id"].map(rows.item2idx).to_numpy()
    cat, name, attr = (df[c].to_numpy(dtype=object) for c in TEXT_COLUMNS)

    def record(i):
        return (int(item[i]), cat[i], name[i], None if pd.isna(attr[i]) else attr[i])

    out = {}
    for uid, idx in zip(uniques.tolist(), by_user):
        last = paid[idx].max()
        hist = idx[paid[idx] != last]
        hist = hist[np.argsort(paid[hist], kind="stable")]             # time order; rows of one payment time stay in file order
        orders = []
        for i in hist:
            if not orders or orders[-1][0] != int(paid[i]):
                orders.append((int(paid[i]), []))
            orders[-1][1].append(record(i))
        out[str(rows.user2idx[int(uid)])] = UserRows(int(uid), orders, record(idx[paid[idx] == last][0]))
    return out


def match(ds, rows):
    """The rows must tell what the data folder tells: the same numbering of items and users, the same users in each split, for every
    user the same orders (as sets of items, in time order) and the same target. Returns the list of differences (empty = identical)."""
    diff = []
    if ds.num_items != len(rows.item2idx):
        diff.append(f"{ds.num_items} items in the data folder, {len(rows.item2idx)} in the split files")
    if ds.item2idx is not None and ds.item2idx != rows.item2idx:
        diff.append("item2idx.json differs from the item numbering of the split files")
    if ds.user2idx is not None and ds.user2idx != rows.user2idx:
        diff.append("user2idx.json differs from the user numbering of the split files")
    for split in SPLITS:
        a, b = getattr(ds, split), rows.users[split]
        if set(a.users) != set(b):
            diff.append(f"{split}: {len(set(a.users) ^ set(b))} users are in only one of the two inputs")
            continue
        targets = sum(a.target[u] != b[u].target[0] for u in a.users)
        orders = sum(a.orders[u] != [sorted({r[0] for r in o}) for _, o in b[u].orders][-MAX_ORDERS:] for u in a.users)
        if targets:
            diff.append(f"{split}: the target differs for {targets} users")
        if orders:
            diff.append(f"{split}: the orders differ for {orders} users")
    return diff


def describe(ds, rows=None):
    """counts that identify the data; printed at the start of a run and stored with the results"""
    out = {
        "num_items": ds.num_items,
        "users_train": len(ds.train.users), "users_val": len(ds.val.users), "users_test": len(ds.test.users),
        "train_orders": int(sum(len(o) for o in ds.train.orders.values())),
        "train_history_items": int(sum(len(x) for o in ds.train.orders.values() for x in o)),
        "max_orders_of_a_user": int(max(len(o) for s in (ds.train, ds.val, ds.test) for o in s.orders.values())),
        "test_repeat_share": float(np.mean([ds.test.target[u] in ds.test.seen[u] for u in ds.test.users])),
    }
    if rows is not None:
        out.update({f"rows_{s}": int(len(rows.frames[s])) for s in SPLITS})
        out["history_rows_test"] = int(sum(len(o) for u in rows.users["test"].values() for _, o in u.orders))
    return out
