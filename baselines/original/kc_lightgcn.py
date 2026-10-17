"""Baseline of the paper (Section 5.2; Table 2, Tables A1 and A3): LightGCN (mean-pool history, dot score)
Dataset: Katcher's
Run: python baselines/original/kc_lightgcn.py --data-dir <DATA_DIR> --emb-dir <EMB_DIR> --out-dir <OUT_DIR> (the paper's numbers: scripts/09_baselines_full.sh)
"""
import argparse as _argparse
import os as _os


def _parse_args():
    p = _argparse.ArgumentParser(description=__doc__, formatter_class=_argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="directory with the dataset json/csv files")
    p.add_argument("--emb-dir", default=None, help="directory with the *.npy embeddings (default: --data-dir)")
    p.add_argument("--out-dir", default="results/katchers/baseline_lightgcn", help="results / checkpoints are written here")
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

DATA_DIR = ARGS.data_dir
EMB_DIR = ARGS.emb_dir
GRAPH_EMB_TEMPLATE = os.path.join(
    EMB_DIR,
    "item_embeddings_graph_seed{seed}.npy",
)


TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.json")
VAL_HISTORY_PATH = os.path.join(DATA_DIR, "val_history.json")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val_data.json")
TEST_HISTORY_PATH = os.path.join(DATA_DIR, "test_history.json")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.json")
NEGATIVE_TEST_PATH = os.path.join(DATA_DIR, "negative_test.json")

MAX_ORDERS = 50

RESULT_DIR = Path(ARGS.out_dir)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = (
    RESULT_DIR
    / "katcher_lightgcn_samegraph_5seed_novel_repeat_summary.json"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)
print("seeds:", SEEDS)
print("graph source:", GRAPH_EMB_TEMPLATE)
print("NOTE: no GRAPH_FREQ in this corrected Katcher baseline")

# ===== 1. Load data + validate the SAME graph embeddings used by the main model =====
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

required = [
    TRAIN_DATA_PATH,
    VAL_HISTORY_PATH,
    VAL_DATA_PATH,
    TEST_HISTORY_PATH,
    TEST_DATA_PATH,
    NEGATIVE_TEST_PATH,
] + [
    GRAPH_EMB_TEMPLATE.format(seed=s)
    for s in SEEDS
]

missing = [p for p in required if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        "필수 파일이 없습니다:\n- "
        + "\n- ".join(missing)
    )

train_data = load_json(TRAIN_DATA_PATH)
val_history = load_json(VAL_HISTORY_PATH)
val_data = load_json(VAL_DATA_PATH)
test_history = load_json(TEST_HISTORY_PATH)
test_data = load_json(TEST_DATA_PATH)
negative_test_dict = load_json(NEGATIVE_TEST_PATH)


first_graph = np.load(
    GRAPH_EMB_TEMPLATE.format(seed=SEEDS[0]),
    mmap_mode="r",
)
NUM_ITEMS = int(first_graph.shape[0])
GRAPH_DIM = int(first_graph.shape[1])

for seed in SEEDS:
    path = GRAPH_EMB_TEMPLATE.format(seed=seed)
    arr = np.load(path, mmap_mode="r")

    if arr.shape != (NUM_ITEMS, GRAPH_DIM):
        raise ValueError(
            f"seed {seed} graph shape={arr.shape}, "
            f"expected=({NUM_ITEMS}, {GRAPH_DIM})"
        )

    print(
        f"seed {seed} graph OK:",
        arr.shape,
        "|",
        path,
    )

print("NUM_ITEMS:", NUM_ITEMS)
print("GRAPH_DIM:", GRAPH_DIM)
print(
    "split train/val/test:",
    len(train_data),
    len(val_history),
    len(test_history),
)

# ===== 2. Exact Katcher history preprocessing from the main model =====
def sort_events(events):
    return sorted(
        events,
        key=lambda event: (
            event.get("day", 0),
            str(event.get("order_id", "")),
        ),
    )

def flatten_history_exact(events):
    """
    Same filtering as the main Katcher model:
      1) chronological sort
      2) retain last MAX_ORDERS
      3) sort unique valid items per transaction
      4) flatten retained transactions
    """
    recent_events = sort_events(events)[-MAX_ORDERS:]

    item_ids = []

    for event in recent_events:
        valid_items = sorted({
            int(item)
            for item in event.get("items", [])
            if 0 <= int(item) < NUM_ITEMS
        })

        if not valid_items:
            continue

        item_ids.extend(valid_items)

    return item_ids

VAL_HISTORY_ITEMS = {
    str(uid): flatten_history_exact(events)
    for uid, events in val_history.items()
}

TEST_HISTORY_ITEMS = {
    str(uid): flatten_history_exact(events)
    for uid, events in test_history.items()
}

def target_to_int(v):
    if isinstance(v, dict):
        items = v.get("items", [])
        if not items:
            raise ValueError(
                f"target dict has empty items: {v}"
            )
        return int(items[0])

    if isinstance(v, list):
        if not v:
            raise ValueError("empty target list")
        return int(v[0])

    return int(v)

VAL_TARGETS = {
    str(uid): target_to_int(v)
    for uid, v in val_data.items()
}

TEST_TARGETS = {
    str(uid): target_to_int(v)
    for uid, v in test_data.items()
}

def normalize_negs(v):
    if isinstance(v, dict):
        v = v.get("negs", [])
    return [int(x) for x in v]

TEST_NEGATIVES = {
    str(uid): normalize_negs(v)
    for uid, v in negative_test_dict.items()
}

# Novel / Repeat classification uses FULL observed pre-target history,
# exactly as in the main experiment.
def full_seen(events):
    seen = set()

    for event in events:
        for item in event.get("items", []):
            item = int(item)
            if 0 <= item < NUM_ITEMS:
                seen.add(item)

    return seen

TEST_SEEN_FULL = {
    str(uid): full_seen(events)
    for uid, events in test_history.items()
}

novel_n = 0
repeat_n = 0

for uid, target in TEST_TARGETS.items():
    if target in TEST_SEEN_FULL.get(uid, set()):
        repeat_n += 1
    else:
        novel_n += 1

print(
    "Novel/Repeat:",
    f"novel={novel_n}",
    f"repeat={repeat_n}",
    f"total={novel_n + repeat_n}",
)
print(
    "empty exact-history test:",
    sum(
        len(v) == 0
        for v in TEST_HISTORY_ITEMS.values()
    ),
)

# ===== 3. Fixed 1:49 candidate protocol =====
TEST_CANDIDATES = {}

for uid, target in TEST_TARGETS.items():
    if uid not in TEST_HISTORY_ITEMS:
        continue

    if len(TEST_HISTORY_ITEMS[uid]) == 0:
        continue

    if uid not in TEST_NEGATIVES:
        continue

    negs = [
        int(item)
        for item in TEST_NEGATIVES[uid]
        if (
            0 <= int(item) < NUM_ITEMS
            and int(item) != target
        )
    ]

    # Preserve order, remove duplicates.
    negs = list(dict.fromkeys(negs))

    if len(negs) < 49:
        raise ValueError(
            f"uid={uid}: valid negatives="
            f"{len(negs)} < 49"
        )

    TEST_CANDIDATES[uid] = [
        int(target),
        *negs[:49],
    ]

print(
    "fixed test candidates:",
    len(TEST_CANDIDATES),
)

if len(TEST_CANDIDATES) != len(TEST_TARGETS):
    print(
        "주의: target users와 evaluable users 수가 다릅니다.",
        len(TEST_TARGETS),
        "->",
        len(TEST_CANDIDATES),
    )

# ===== 4. LightGCN co-purchase baseline scoring =====
# No additional recommender training.
#
# u = mean(history item graph embeddings)
# score(u, c) = u^T e_c
#
# These graph embeddings are exactly the precomputed LightGCN embeddings
# loaded by the Katcher main model for each seed.

@torch.no_grad()
def evaluate_seed(seed):
    graph = np.load(
        GRAPH_EMB_TEMPLATE.format(seed=seed)
    )

    graph_emb = torch.tensor(
        graph,
        dtype=torch.float32,
        device=device,
    )

    vals = {
        group: {
            5: [],
            10: [],
        }
        for group in (
            "overall",
            "novel",
            "repeat",
        )
    }

    per_user = {}

    for uid, candidates in TEST_CANDIDATES.items():
        history_items = TEST_HISTORY_ITEMS[uid]

        hist = torch.tensor(
            history_items,
            dtype=torch.long,
            device=device,
        )

        candidate_tensor = torch.tensor(
            candidates,
            dtype=torch.long,
            device=device,
        )

        user_repr = graph_emb[hist].mean(dim=0)

        scores = torch.matmul(
            graph_emb[candidate_tensor],
            user_repr,
        )

        ranking = torch.argsort(
            scores,
            descending=True,
        )

        # Target is local candidate index 0.
        rank = int(
            (ranking == 0)
            .nonzero(as_tuple=False)[0, 0]
            .item()
        )

        target = int(candidates[0])

        group = (
            "repeat"
            if target in TEST_SEEN_FULL[uid]
            else "novel"
        )

        ndcg5 = (
            1.0 / np.log2(rank + 2)
            if rank < 5
            else 0.0
        )

        ndcg10 = (
            1.0 / np.log2(rank + 2)
            if rank < 10
            else 0.0
        )

        vals["overall"][5].append(float(ndcg5))
        vals["overall"][10].append(float(ndcg10))
        vals[group][5].append(float(ndcg5))
        vals[group][10].append(float(ndcg10))

        per_user[uid] = {
            "group": group,
            "ndcg@5": float(ndcg5),
            "ndcg@10": float(ndcg10),
        }

    summary = {}

    for group in (
        "overall",
        "novel",
        "repeat",
    ):
        summary[group] = {
            "n": len(vals[group][10]),
            "ndcg@5": float(
                np.mean(vals[group][5])
            ),
            "ndcg@10": float(
                np.mean(vals[group][10])
            ),
        }

    del graph_emb

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary, per_user

# ===== 5. Run seeds 42..46 =====
all_seed_results = []

for seed in SEEDS:
    print("\n" + "=" * 90)
    print(
        f"KATCHER | SAME-GRAPH LightGCN BASELINE | "
        f"SEED={seed}"
    )
    print("=" * 90)

    grouped, per_user = evaluate_seed(seed)

    for group in (
        "overall",
        "novel",
        "repeat",
    ):
        row = grouped[group]
        print(
            f"{group:7s} | "
            f"NDCG@5={row['ndcg@5']:.4f} | "
            f"NDCG@10={row['ndcg@10']:.4f} | "
            f"n={row['n']}"
        )

    per_user_path = (
        RESULT_DIR
        / f"katcher_lightgcn_samegraph_seed{seed}_per_user.json"
    )

    with per_user_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            per_user,
            f,
            ensure_ascii=False,
            indent=2,
        )

    all_seed_results.append({
        "seed": int(seed),
        "graph_embedding": (
            GRAPH_EMB_TEMPLATE.format(seed=seed)
        ),
        "test": grouped,
        "per_user_path": str(per_user_path),
    })

print("\nall seeds finished")

# ===== 6. 5-seed mean ± std =====
aggregate = {}

for group in (
    "overall",
    "novel",
    "repeat",
):
    aggregate[group] = {}

    for metric in (
        "ndcg@5",
        "ndcg@10",
    ):
        arr = np.array(
            [
                r["test"][group][metric]
                for r in all_seed_results
            ],
            dtype=np.float64,
        )

        aggregate[group][metric] = {
            "mean": float(arr.mean()),
            # Match the main 5-seed notebook's std convention.
            "std": float(arr.std(ddof=1)),
            "seed_values": {
                str(r["seed"]): float(
                    r["test"][group][metric]
                )
                for r in all_seed_results
            },
        }

    aggregate[group]["n"] = int(
        all_seed_results[0]["test"][group]["n"]
    )

final = {
    "dataset": "Katcher",
    "model": (
        "LightGCN co-purchase baseline "
        "(same precomputed graph embeddings as main model)"
    ),
    "training_objective": "BPR during upstream item-item co-purchase LightGCN embedding training; no downstream retraining",
    "seeds": SEEDS,
    "graph_source": (
        "item_embeddings_graph_seed{seed}.npy"
    ),
    "graph_rebuilt_in_notebook": False,
    "graph_freq_parameter": None,
    "scoring_rule": (
        "mean observed-history graph embedding "
        "dot candidate graph embedding"
    ),
    "protocol": (
        "same fixed 1 positive + 49 negatives"
    ),
    "subset_definition": (
        "target membership in full observed "
        "pre-target history"
    ),
    "per_seed": all_seed_results,
    "aggregate": aggregate,
}

with SUMMARY_PATH.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        final,
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\n" + "=" * 90)
print("5-SEED FINAL SUMMARY")
print("=" * 90)

for group in (
    "overall",
    "novel",
    "repeat",
):
    m5 = aggregate[group]["ndcg@5"]
    m10 = aggregate[group]["ndcg@10"]

    print(
        f"{group:7s} | "
        f"NDCG@5="
        f"{m5['mean']:.4f} ± {m5['std']:.4f} | "
        f"NDCG@10="
        f"{m10['mean']:.4f} ± {m10['std']:.4f} | "
        f"n={aggregate[group]['n']}"
    )

print("saved:", SUMMARY_PATH)

