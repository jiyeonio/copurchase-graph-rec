"""Small random data in the file formats of the paper, for checking that the code runs. Not the data of the paper: the numbers it
produces mean nothing.

Writes split files (train/val/test_data.csv with the columns preprocess/katchers.py requires) and runs preprocess/katchers.py on
them, unchanged, to obtain the data folder. So the two inputs of this prototype are made the way the real ones are.

usage: python proto_bert_nsp/make_fake_data.py --out <folder>     -> <folder>/split, <folder>/kc, <folder>/kc_emb
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--users", type=int, default=300)
p.add_argument("--items", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
A = p.parse_args()
rng = np.random.default_rng(A.seed)
N, CLUSTER = A.items, 20
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORDS = ["사과", "배", "달걀", "우유", "두부", "김치", "라면", "생수", "휴지", "세제", "샴푸", "양말", "수건", "커피", "과자", "만두"]
CATS = ["과일", "유제품", "반찬", "면류", "생활용품", "음료", "간식", "냉동식품"]
names = [f"{WORDS[i % len(WORDS)]} {WORDS[(i * 7 + 3) % len(WORDS)]} {100 + i}g" for i in range(N)]
cats = [CATS[i % len(CATS)] for i in range(N)]
T0 = 1_680_000_000_000                                   # payment times are milliseconds, as in the purchase log


def name_of(i):
    """a product is sold under a tagged name most of the time and under its plain name otherwise"""
    return f"[특가] {names[i]}" if rng.random() < 0.6 else names[i]


def attr_of():
    return float("nan") if rng.random() < 0.3 else f"{int(rng.integers(1, 6))}개,선택{int(rng.integers(1, 4))}"


def user_rows(uid, n_orders, max_size):
    """rows of one user: n_orders orders and one target row at the last payment time; most items come from the user's own group"""
    own = np.arange((uid % (N // CLUSTER)) * CLUSTER, (uid % (N // CLUSTER) + 1) * CLUSTER)
    rows, t = [], T0 + int(rng.integers(0, 10**9))
    for j in range(n_orders + 1):
        t += int(rng.integers(10**6, 10**8))
        k = 1 if j == n_orders else int(rng.integers(1, max_size + 1))
        items = [int(i) for i in (rng.choice(own, min(k, CLUSTER), replace=False) if rng.random() < 0.9 else rng.choice(N, k, replace=False))]
        if j < n_orders and rng.random() < 0.2:
            items.append(items[0])                       # the same product twice in one order, with other attributes
        for i in items:
            rows.append({"user_id": uid, "order_id": f"o{uid}_{j}", "product_id": 1000 + i, "initial_paid_at": float(t),
                         "product_name": name_of(i), "root_category_name": "식품", "category_name": cats[i], "attribute_values": attr_of()})
    return rows


uids = rng.permutation(np.arange(10_000, 10_000 + A.users))
cut = [int(0.6 * A.users), int(0.8 * A.users)]
parts = {"train": uids[:cut[0]], "val": uids[cut[0]:cut[1]], "test": uids[cut[1]:]}
split_dir, data_dir = os.path.join(A.out, "split"), os.path.join(A.out, "kc")
os.makedirs(split_dir, exist_ok=True)
for part, us in parts.items():
    rows = []
    for n, u in enumerate(us):
        long_history = n % 25 == 0                       # a few histories that do not fit into 512 tokens
        rows += user_rows(int(u), int(rng.integers(30, 45)) if long_history else int(rng.integers(1, 7)), 3)
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=A.seed).reset_index(drop=True)      # rows are not stored in time order
    df.loc[len(df)] = {**rows[0], "initial_paid_at": float("nan")}                           # a row without payment time is dropped
    df.to_csv(os.path.join(split_dir, f"{part}_data.csv"), index=False, encoding="utf-8-sig")

r = subprocess.run([sys.executable, os.path.join(ROOT, "preprocess", "katchers.py"), "--data-dir", split_dir, "--output-dir", data_dir],
                   capture_output=True, text=True)
if r.returncode:
    sys.exit("preprocess/katchers.py failed on the fake split files:\n" + r.stdout[-2000:] + r.stderr[-2000:])
os.makedirs(os.path.join(A.out, "kc_emb"), exist_ok=True)
n_items = len(json.load(open(os.path.join(data_dir, "item2idx.json"))))                      # the products that occur in the rows
np.save(os.path.join(A.out, "kc_emb", "item_embeddings_text.npy"), np.zeros((n_items, 768), dtype=np.float32))   # the baselines read its size
print(f"fake data in {A.out}: {A.users} users, {n_items} items | " + r.stdout.strip().splitlines()[-3])
