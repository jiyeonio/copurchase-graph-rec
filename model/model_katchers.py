"""Katcher's: data loading for the transaction-level Transformer (model and training loop in core.py).
usage: python model/model_katchers.py --exp NAME --src text+graph --text-path TEXT.npy --graph-dir GRAPH_EMB_DIR --test
"""
import json
import os
import sys
import zlib

import numpy as np
import torch

import core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import KC

VAL_NEGATIVE_SEED = 42
args = core.parse_args(KC, eval_every=2, val_batch=256, text_required=True)
D = args.data_dir


def load_json(name):
    with open(os.path.join(D, name), "r", encoding="utf-8") as f:
        return json.load(f)


train_data = load_json("train_data.json")
train_targets = load_json("train_targets.json")
val_data = load_json("val_data.json")
val_history = load_json("val_history.json")
text = torch.as_tensor(np.load(args.text_path), dtype=torch.float32, device=core.device)
NUM_ITEMS = int(text.shape[0])


def flatten(events):
    return core.flatten_history(events, NUM_ITEMS, key=lambda e: (e.get("day", 0), str(e.get("order_id", ""))))


def seen_items(history, uids):
    return {u: {int(i) for e in history.get(u, []) for i in e.get("items", []) if 0 <= int(i) < NUM_ITEMS} for u in uids}


def val_negatives(uid, target, n=49):
    pool = np.array([i for i in range(NUM_ITEMS) if i != target], dtype=np.int64)
    rng = np.random.default_rng((VAL_NEGATIVE_SEED + zlib.crc32(str(uid).encode("utf-8"))) % (2**32))
    return rng.choice(pool, size=n, replace=False).tolist()


train_cache = core.build_cache(train_data, flatten)
val_cache = core.build_cache(val_history, flatten)
val_candidates = {}
for uid, ev in val_data.items():
    uid = str(uid)
    ti = ev.get("items", [])
    if uid not in val_cache or not ti or not (0 <= int(ti[0]) < NUM_ITEMS):
        continue
    val_candidates[uid] = [int(ti[0]), *val_negatives(uid, int(ti[0]), 49)]

test_candidates, test_cache, test_data, test_history = {}, {}, {}, {}
if args.test:                                   # test negatives: the fixed file of the dataset
    test_data = load_json("test_data.json")
    test_history = load_json("test_history.json")
    neg = load_json("negative_test.json")
    test_cache = core.build_cache(test_history, flatten)
    for uid, ev in test_data.items():
        uid = str(uid)
        ti = ev.get("items", [])
        if uid not in test_cache or not ti or not (0 <= int(ti[0]) < NUM_ITEMS) or uid not in neg:
            continue
        t = int(ti[0])
        ng = list(dict.fromkeys(int(i) for i in neg[uid] if 0 <= int(i) < NUM_ITEMS and int(i) != t))
        assert len(ng) >= 49
        test_candidates[uid] = [t, *ng[:49]]

data = core.Data(num_items=NUM_ITEMS, text=text, train_uids=list(train_data.keys()), train_cache=train_cache, train_targets=train_targets,
                 val_cache=val_cache, val_targets=val_data, val_candidates=val_candidates, val_seen=seen_items(val_history, val_candidates),
                 test_cache=test_cache, test_targets=test_data, test_candidates=test_candidates, test_seen=seen_items(test_history, test_candidates))
core.run(args, data, lambda d, seed: f"{d}/item_embeddings_graph_seed{seed}.npy")
