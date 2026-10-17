"""Instacart: data loading for the transaction-level Transformer (model and training loop in core.py).
usage: python model/model_instacart.py --exp NAME --src text+graph --graph-dir GRAPH_EMB_DIR --test
"""
import json
import os
import sys
import zlib

import numpy as np
import torch

import core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import IC

VAL_NEGATIVE_SEED, GRAPH_FREQ = 42, 3
args = core.parse_args(IC, eval_every=4, val_batch=64, text_required=False)
D = args.data_dir


def load_json(name):
    with open(os.path.join(D, name), "r", encoding="utf-8") as f:
        return json.load(f)


split = load_json("split.json")
target = load_json("target.json")
user_item_history = load_json("user_item_history.json")
NUM_ITEMS = len(load_json("item_meta.json"))
item_text = np.load(args.text_path or os.path.join(D, "item_embeddings_text.npy"))
assert item_text.shape == (NUM_ITEMS, core.TEXT_RAW_DIM), item_text.shape
text = torch.as_tensor(item_text, dtype=torch.float32, device=core.device)
train_users, val_users, test_users = split["train"], split["val"], split["test"]
assert not (set(map(str, train_users)) & (set(map(str, val_users)) | set(map(str, test_users))))


def build_events(uid):
    ev = []
    for position, items in enumerate(user_item_history[str(uid)]):
        valid = sorted({int(i) for i in items if 0 <= int(i) < NUM_ITEMS})
        if valid:
            ev.append({"items": valid, "day": position, "order_idx": position})
    return ev


def flatten(events):
    return core.flatten_history(events, NUM_ITEMS)


def targets(users):
    return {str(u): {"items": [int(target[str(u)])]} for u in users}


def seen_items(history, uids):
    return {u: {int(i) for e in history[u] for i in e["items"]} for u in uids}


def val_negatives(uid, events, target_item, n=49):
    """validation negatives: 49 items outside the user's history, fixed per user"""
    seen = {int(i) for e in events for i in e["items"]}
    pool = np.array(sorted(set(range(NUM_ITEMS)) - seen - {target_item}), dtype=np.int64)
    if len(pool) < n:
        return None
    rng = np.random.default_rng((VAL_NEGATIVE_SEED + zlib.crc32(uid.encode("utf-8")) + GRAPH_FREQ * 1000) % (2**32))
    return rng.choice(pool, size=n, replace=False).tolist()


train_data = {str(u): build_events(u) for u in train_users}
val_history = {str(u): build_events(u) for u in val_users}
val_data = targets(val_users)
train_cache = core.build_cache(train_data, flatten)
val_cache = core.build_cache(val_history, flatten)
val_candidates = {}
for uid, ev in val_data.items():
    if uid not in val_cache:
        continue
    t = int(ev["items"][0])
    negs = val_negatives(uid, val_history[uid], t, 49)
    if negs is not None:
        val_candidates[uid] = [t, *negs]

test_candidates, test_cache, test_data, test_history = {}, {}, {}, {}
if args.test:                                   # test negatives: the fixed file of the dataset
    negative = load_json("negative_test.json")
    test_history = {str(u): build_events(u) for u in test_users}
    test_data = targets(test_users)
    test_cache = core.build_cache(test_history, flatten)
    for uid, ev in test_data.items():
        if uid not in test_cache or uid not in negative:
            continue
        t = int(ev["items"][0])
        negs = [int(x) for x in negative[uid]["negs"] if int(x) != t]
        if len(negs) >= 49:
            test_candidates[uid] = [t, *negs[:49]]
print(f"Instacart: items {NUM_ITEMS} | train/val users {len(train_cache)}/{len(val_cache)} | val candidates {len(val_candidates)} | test candidates {len(test_candidates)}", flush=True)

data = core.Data(num_items=NUM_ITEMS, text=text, train_uids=list(train_data.keys()), train_cache=train_cache, train_targets=targets(train_users),
                 val_cache=val_cache, val_targets=val_data, val_candidates=val_candidates, val_seen=seen_items(val_history, val_candidates),
                 test_cache=test_cache, test_targets=test_data, test_candidates=test_candidates, test_seen=seen_items(test_history, test_candidates))
core.run(args, data, lambda d, seed: f"{d}/item_embeddings_graph_freq{GRAPH_FREQ}_seed{seed}.npy")
