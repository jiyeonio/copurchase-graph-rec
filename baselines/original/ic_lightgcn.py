"""Baseline of the paper (Section 5.2; Table 2, Tables A1 and A3): LightGCN (mean-pool history, dot score)
Dataset: Instacart
Run: python baselines/original/ic_lightgcn.py --data-dir <DATA_DIR> --emb-dir <EMB_DIR> --out-dir <OUT_DIR> (the paper's numbers: scripts/09_baselines_full.sh)
"""
import argparse as _argparse
import os as _os


def _parse_args():
    p = _argparse.ArgumentParser(description=__doc__, formatter_class=_argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="directory with the dataset json/csv files")
    p.add_argument("--emb-dir", default=None, help="directory with the *.npy embeddings (default: --data-dir)")
    p.add_argument("--out-dir", default="results/instacart/baseline_lightgcn", help="results / checkpoints are written here")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="random seeds")
    return p.parse_args()


ARGS = _parse_args()
if ARGS.emb_dir is None:
    ARGS.emb_dir = ARGS.data_dir
_os.makedirs(ARGS.out_dir, exist_ok=True)
RESULT_ROOT = ARGS.out_dir
CKPT_DIR = _os.path.join(ARGS.out_dir, "checkpoints")
_os.makedirs(CKPT_DIR, exist_ok=True)



# ===== 0. Config =====
import os
import json
from pathlib import Path
import numpy as np
import torch

SEEDS = ARGS.seeds
GRAPH_FREQ = 3
DATA_DIR = ARGS.data_dir
EMB_DIR = ARGS.emb_dir
def graph_embedding_path(seed):
    candidates = [
        os.path.join(EMB_DIR, f"item_embeddings_graph_freq{GRAPH_FREQ}_seed{seed}.npy"),
    ]
    if GRAPH_FREQ == 1:
        candidates.append(os.path.join(EMB_DIR, f"item_embeddings_graph_seed{seed}.npy"))
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"graph embedding missing for seed={seed}: {candidates}")

SPLIT_PATH = os.path.join(DATA_DIR, "split.json")
TARGET_PATH = os.path.join(DATA_DIR, "target.json")
NEGATIVE_TEST_PATH = os.path.join(DATA_DIR, "negative_test.json")
ITEM_META_PATH = os.path.join(DATA_DIR, "item_meta.json")
USER_ITEM_HISTORY_PATH = os.path.join(DATA_DIR, "user_item_history.json")

MAX_ORDERS = 50
RESULT_DIR = Path(ARGS.out_dir)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = RESULT_DIR / f"instacart_lightgcn_samegraph_freq{GRAPH_FREQ}_5seed.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("GRAPH_FREQ:", GRAPH_FREQ)



# ===== 1. Load exact Instacart LF split / target / graph embeddings =====
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

required = [
    SPLIT_PATH, TARGET_PATH, NEGATIVE_TEST_PATH,
    ITEM_META_PATH, USER_ITEM_HISTORY_PATH,
] + [graph_embedding_path(s) for s in SEEDS]

missing = [p for p in required if not os.path.exists(p)]
if missing:
    raise FileNotFoundError("missing files:\n- " + "\n- ".join(missing))

split = load_json(SPLIT_PATH)
target = load_json(TARGET_PATH)
negative_raw = load_json(NEGATIVE_TEST_PATH)
item_meta = load_json(ITEM_META_PATH)
user_item_history = load_json(USER_ITEM_HISTORY_PATH)

train_users = [str(x) for x in split["train"]]
val_users = [str(x) for x in split["val"]]
test_users = [str(x) for x in split["test"]]
NUM_ITEMS = len(item_meta)

test_targets = {u: int(target[u]) for u in test_users}
negative_test_dict = {
    u: [int(x) for x in negative_raw[u]["negs"]]
    for u in test_users
}

first_graph = np.load(graph_embedding_path(SEEDS[0]), mmap_mode="r")
GRAPH_DIM = int(first_graph.shape[1])
if first_graph.shape[0] != NUM_ITEMS:
    raise ValueError(f"graph items={first_graph.shape[0]} != NUM_ITEMS={NUM_ITEMS}")

for seed in SEEDS:
    arr = np.load(graph_embedding_path(seed), mmap_mode="r")
    if arr.shape != (NUM_ITEMS, GRAPH_DIM):
        raise ValueError(f"seed={seed} graph shape={arr.shape}")

print("NUM_ITEMS:", NUM_ITEMS)
print("GRAPH_DIM:", GRAPH_DIM)
print("split train/val/test:", len(train_users), len(val_users), len(test_users))



# ===== 2. Exact Instacart LF item-history contract =====
def build_history_items(uid):
    seq = []
    for order_items in user_item_history[str(uid)][-MAX_ORDERS:]:
        valid = sorted({
            int(x)
            for x in order_items
            if 0 <= int(x) < NUM_ITEMS
        })
        seq.extend(valid)
    return seq

TEST_HISTORY_ITEMS = {
    u: build_history_items(u)
    for u in test_users
}

# Novel/Repeat uses FULL observed pre-target history, same as LF.
TEST_SEEN_FULL = {
    u: {
        int(x)
        for order in user_item_history[u]
        for x in order
        if 0 <= int(x) < NUM_ITEMS
    }
    for u in test_users
}

TEST_CANDIDATES = {}
for u in test_users:
    hist = TEST_HISTORY_ITEMS.get(u, [])
    if not hist or u not in negative_test_dict:
        continue
    t = int(test_targets[u])
    negs = [int(x) for x in negative_test_dict[u] if int(x) != t]
    if len(negs) < 49:
        continue
    TEST_CANDIDATES[u] = [t, *negs[:49]]

print("fixed test candidates:", len(TEST_CANDIDATES))



# ===== 3. Same-graph LightGCN scoring =====
@torch.no_grad()
def evaluate_seed(seed):
    graph_emb = torch.tensor(
        np.load(graph_embedding_path(seed)),
        dtype=torch.float32,
        device=device,
    )

    vals = {g: {5: [], 10: []} for g in ("overall", "novel", "repeat")}
    per_user = {}

    for uid, candidates in TEST_CANDIDATES.items():
        hist = torch.tensor(TEST_HISTORY_ITEMS[uid], dtype=torch.long, device=device)
        cand = torch.tensor(candidates, dtype=torch.long, device=device)

        user_repr = graph_emb[hist].mean(dim=0)
        scores = torch.matmul(graph_emb[cand], user_repr)
        ranking = torch.argsort(scores, descending=True)
        rank = int((ranking == 0).nonzero(as_tuple=False)[0, 0].item())

        target = int(candidates[0])
        group = "repeat" if target in TEST_SEEN_FULL[uid] else "novel"

        rec = {"group": group}
        for k in (5, 10):
            v = 1.0 / np.log2(rank + 2) if rank < k else 0.0
            vals["overall"][k].append(float(v))
            vals[group][k].append(float(v))
            rec[f"ndcg@{k}"] = float(v)
        per_user[uid] = rec

    summary = {}
    for g in ("overall", "novel", "repeat"):
        summary[g] = {
            "n": len(vals[g][10]),
            "ndcg@5": float(np.mean(vals[g][5])) if vals[g][5] else None,
            "ndcg@10": float(np.mean(vals[g][10])) if vals[g][10] else None,
        }

    del graph_emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, per_user



# ===== 4. Run seeds 42..46 =====
all_seed_results = []

for seed in SEEDS:
    print("\n" + "=" * 90)
    print(f"INSTACART | SAME-GRAPH LightGCN | seed={seed}")
    print("=" * 90)

    grouped, per_user = evaluate_seed(seed)

    for g in ("overall", "novel", "repeat"):
        row = grouped[g]
        print(
            f"{g:7s} | NDCG@5={row['ndcg@5']:.4f} | "
            f"NDCG@10={row['ndcg@10']:.4f} | n={row['n']}"
        )

    per_user_path = RESULT_DIR / f"instacart_lightgcn_samegraph_seed{seed}_per_user.json"
    with per_user_path.open("w", encoding="utf-8") as f:
        json.dump(per_user, f, ensure_ascii=False, indent=2)

    all_seed_results.append({
        "seed": int(seed),
        "graph_embedding": graph_embedding_path(seed),
        "test": grouped,
        "per_user_path": str(per_user_path),
    })



# ===== 5. Five-seed mean ± sample std =====
aggregate = {}

for g in ("overall", "novel", "repeat"):
    aggregate[g] = {}
    for metric in ("ndcg@5", "ndcg@10"):
        arr = np.asarray(
            [r["test"][g][metric] for r in all_seed_results],
            dtype=np.float64,
        )
        aggregate[g][metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "per_seed": {
                str(r["seed"]): float(r["test"][g][metric])
                for r in all_seed_results
            },
        }
    aggregate[g]["n"] = int(all_seed_results[0]["test"][g]["n"])

final = {
    "dataset": "Instacart",
    "model": "LightGCN same-graph embedding baseline",
    "training_objective": "BPR during upstream item-item co-purchase LightGCN embedding training; no downstream retraining",
    "seeds": SEEDS,
    "graph_freq": GRAPH_FREQ,
    "graph_source": f"item_embeddings_graph_freq{GRAPH_FREQ}_seed{{seed}}.npy",
    "scoring_rule": "mean LF-matched observed-history graph embedding dot candidate graph embedding",
    "protocol": "Instacart LF-matched history + fixed 1 positive : 49 negatives",
    "subset_definition": "target membership in FULL observed pre-target history",
    "aggregate": aggregate,
    "per_seed": all_seed_results,
}

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

for g in ("overall", "novel", "repeat"):
    m5 = aggregate[g]["ndcg@5"]
    m10 = aggregate[g]["ndcg@10"]
    print(
        f"{g:7s} | "
        f"NDCG@5={m5['mean']:.4f} ± {m5['std']:.4f} | "
        f"NDCG@10={m10['mean']:.4f} ± {m10['std']:.4f} | "
        f"n={aggregate[g]['n']}"
    )
print("saved:", SUMMARY_PATH)

