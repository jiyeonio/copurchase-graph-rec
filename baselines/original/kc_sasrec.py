"""Baseline of the paper (Section 5.2; Table 2, Tables A1 and A3): SASRec (BCE + 1 sampled negative)
Dataset: Katcher's
Run: python baselines/original/kc_sasrec.py --data-dir <DATA_DIR> --emb-dir <EMB_DIR> --out-dir <OUT_DIR> (the paper's numbers: scripts/09_baselines_full.sh)
"""
import argparse as _argparse
import os as _os


def _parse_args():
    p = _argparse.ArgumentParser(description=__doc__, formatter_class=_argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="directory with the dataset json/csv files")
    p.add_argument("--emb-dir", default=None, help="directory with the *.npy embeddings (default: --data-dir)")
    p.add_argument("--out-dir", default="results/katchers/baseline_sasrec", help="results / checkpoints are written here")
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
import random
import json
import zlib
import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEEDS = ARGS.seeds
SEED = SEEDS[0]
DATA_DIR = ARGS.data_dir
EMB_DIR = ARGS.emb_dir
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.json")
TRAIN_TARGETS_PATH = os.path.join(DATA_DIR, "train_targets.json")
VAL_HISTORY_PATH = os.path.join(DATA_DIR, "val_history.json")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val_data.json")
TEST_HISTORY_PATH = os.path.join(DATA_DIR, "test_history.json")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.json")
NEGATIVE_TEST_PATH = os.path.join(DATA_DIR, "negative_test.json")

# These are only used to reproduce the LF Katcher history-validity contract
# and catalog size. Their embeddings are NOT fed into GRU4Rec/SASRec.
ITEM_EMB_TEXT_PATH = os.path.join(EMB_DIR, "item_embeddings_text.npy")

MAX_ORDERS = 50

HIDDEN_DIM = 64
NUM_BLOCKS = 2
NUM_HEADS = 4
FF_DIM = 128
DROPOUT = 0.1

BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100
EVAL_EVERY = 2
PATIENCE = 5

VAL_NEGATIVE_SEED = 42

BEST_MODEL_PATH = os.path.join(
    CKPT_DIR,
    f"best_katcher_sasrec_original_bce1neg_seed{SEED}.pt",
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== 1. Load exact Katcher split / targets / fixed negatives =====
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

required = [
    TRAIN_DATA_PATH,
    TRAIN_TARGETS_PATH,
    VAL_HISTORY_PATH,
    VAL_DATA_PATH,
    TEST_HISTORY_PATH,
    TEST_DATA_PATH,
    NEGATIVE_TEST_PATH,
    ITEM_EMB_TEXT_PATH,
]
missing = [p for p in required if not os.path.exists(p)]
if missing:
    raise FileNotFoundError("필수 Katcher 파일이 없습니다:\n- " + "\n- ".join(missing))

train_data = load_json(TRAIN_DATA_PATH)
train_targets_raw = load_json(TRAIN_TARGETS_PATH)
val_history = load_json(VAL_HISTORY_PATH)
val_data_raw = load_json(VAL_DATA_PATH)
test_history = load_json(TEST_HISTORY_PATH)
test_data_raw = load_json(TEST_DATA_PATH)
negative_raw = load_json(NEGATIVE_TEST_PATH)

NUM_ITEMS = int(np.load(ITEM_EMB_TEXT_PATH, mmap_mode="r").shape[0])

def _target_to_int(v):
    if isinstance(v, dict):
        items = v.get("items", [])
        if not items:
            raise ValueError(f"target dict에 items가 비어 있습니다: {v}")
        return int(items[0])
    if isinstance(v, list):
        if not v:
            raise ValueError("target list가 비어 있습니다.")
        return int(v[0])
    return int(v)

train_users = [str(uid) for uid in train_data.keys()]
val_users = [str(uid) for uid in val_history.keys()]
test_users = [str(uid) for uid in test_history.keys()]

train_targets = {uid: _target_to_int(train_targets_raw[uid]) for uid in train_users}
val_targets = {uid: _target_to_int(val_data_raw[uid]) for uid in val_users}
test_targets = {uid: _target_to_int(test_data_raw[uid]) for uid in test_users}

def _normalize_negs(v):
    if isinstance(v, dict):
        v = v.get("negs", [])
    return [int(x) for x in v]

negative_test_dict = {
    uid: _normalize_negs(negative_raw[uid])
    for uid in test_users
}

assert set(train_users).isdisjoint(set(val_users))
assert set(train_users).isdisjoint(set(test_users))
assert set(val_users).isdisjoint(set(test_users))

for uid, t in {**train_targets, **val_targets, **test_targets}.items():
    if not 0 <= int(t) < NUM_ITEMS:
        raise ValueError(
            f"target out of range: uid={uid}, target={t}, NUM_ITEMS={NUM_ITEMS}"
        )

def _full_seen_from_events(events):
    seen = set()
    for ev in events:
        for x in ev.get("items", []):
            x = int(x)
            if 0 <= x < NUM_ITEMS:
                seen.add(x)
    return seen

TEST_SEEN_ITEMS = {
    uid: _full_seen_from_events(test_history[uid])
    for uid in test_users
}


# ===== 2. Exact LF Katcher history contract -> item sequences =====
def sort_events(events):
    return sorted(
        events,
        key=lambda e: (e.get("day", 0), str(e.get("order_id", ""))),
    )

def build_item_sequence_from_events(events):
    sequence = []
    for ev in sort_events(events)[-MAX_ORDERS:]:
        valid_items = sorted({
            int(x)
            for x in ev.get("items", [])
            if 0 <= int(x) < NUM_ITEMS
        })
        sequence.extend(valid_items)

    return sequence

train_sequences = {
    uid: build_item_sequence_from_events(train_data[uid])
    for uid in train_users
}
val_sequences = {
    uid: build_item_sequence_from_events(val_history[uid])
    for uid in val_users
}
test_sequences = {
    uid: build_item_sequence_from_events(test_history[uid])
    for uid in test_users
}

MAX_ITEM_SEQ_LEN = max(
    max((len(v) for v in train_sequences.values()), default=1),
    max((len(v) for v in val_sequences.values()), default=1),
    max((len(v) for v in test_sequences.values()), default=1),
)

def _stats(d):
    arr = np.array([len(v) for v in d.values()], dtype=np.int64)
    return {
        "users": int(len(arr)),
        "empty": int((arr == 0).sum()),
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "median": float(np.median(arr)) if len(arr) else 0.0,
        "max": int(arr.max()) if len(arr) else 0,
    }

print("train sequence:", _stats(train_sequences))
print("val sequence:", _stats(val_sequences))
print("test sequence:", _stats(test_sequences))
print("MAX_ITEM_SEQ_LEN from last-50-tx histories:", MAX_ITEM_SEQ_LEN)


# ===== 3. PURE SASRec baseline =====
# Canonical SASRec-style stack only: item+position embedding -> causal self-attention
# -> point-wise FFN with residual/LayerNorm -> item-embedding scoring.
# No graph, memory, fusion, recurrent module, cross-attention, or auxiliary encoder.

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.dropout1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.dropout2(y)
        return y.transpose(1, 2)


class SASRecBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.ffn = PointWiseFeedForward(hidden_dim, dropout)

    def forward(self, x, causal_mask, padding_mask):
        q = self.attn_norm(x)
        attn_out, _ = self.attn(
            q, q, q,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SASRecBaseline(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        max_seq_len: int = 512,
        num_blocks: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_items = int(num_items)
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)

        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len + 1, hidden_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-8)

        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
            self.position_embedding.weight[0].zero_()

    def encode_all(self, input_tokens):
        if input_tokens.ndim != 2:
            raise ValueError("input_tokens must be [B, L]")
        batch_size, seq_len = input_tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} > max_seq_len={self.max_seq_len}")

        positions = torch.arange(1, seq_len + 1, device=input_tokens.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        positions = positions.masked_fill(input_tokens.eq(0), 0)

        x = self.item_embedding(input_tokens) * math.sqrt(self.hidden_dim)
        x = x + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_tokens.device),
            diagonal=1,
        )
        padding_mask = input_tokens.eq(0)

        for block in self.blocks:
            x = block(x, causal_mask, padding_mask)

        x = self.final_norm(x)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x

    def encode(self, input_tokens, lengths):
        if torch.any(lengths <= 0):
            raise ValueError("all sequence lengths must be > 0")
        x = self.encode_all(input_tokens)
        last_idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(input_tokens.size(0), device=input_tokens.device)
        return x[batch_idx, last_idx]

    def score_all_positions(self, input_tokens):
        hidden = self.encode_all(input_tokens)
        return torch.matmul(hidden, self.item_embedding.weight[1:].T)

    def score_all_items(self, input_tokens, lengths):
        u = self.encode(input_tokens, lengths)
        return torch.matmul(u, self.item_embedding.weight[1:].T)

    def score_candidates(self, input_tokens, lengths, candidate_ids):
        if input_tokens.size(0) != 1:
            raise ValueError("score_candidates expects batch_size=1")
        u = self.encode(input_tokens, lengths)[0]
        c = self.item_embedding(candidate_ids + 1)
        return torch.matmul(c, u)


# ===== 4. Sequence-level next-item batch preparation =====
# Original SASRec training objective:
#   - every valid next-item position supplies one positive item
#   - sample ONE negative item for that position
#   - optimize binary cross-entropy on positive/negative logits
#
# Negative sampling follows the canonical SASRec convention:
# sample from the catalog while excluding items appearing in that user's
# observed training sequence (including the held-out train target appended here).

IGNORE_INDEX = -100
SASREC_NUM_NEGATIVES = 1


def _sample_sasrec_negative(
    rng: np.random.Generator,
    forbidden: set,
) -> int:
    if len(forbidden) >= NUM_ITEMS:
        raise RuntimeError(
            "SASRec negative sampling failed: user has no unseen catalog item."
        )

    # Rejection sampling is efficient because each user observes only a small
    # fraction of the full catalog in these datasets.
    while True:
        x = int(rng.integers(0, NUM_ITEMS))
        if x not in forbidden:
            return x


def make_sequence_training_batch(
    user_ids,
    sequence_dict,
    target_dict,
    rng: np.random.Generator,
):
    pairs = []

    for uid in user_ids:
        seq = sequence_dict.get(uid, [])
        tgt = target_dict.get(uid)

        if not seq or tgt is None:
            continue

        tgt = int(tgt)
        if not 0 <= tgt < NUM_ITEMS:
            continue

        full = (
            [int(x) for x in seq]
            + [tgt]
        )[-(MAX_ITEM_SEQ_LEN + 1):]

        if len(full) < 2:
            continue

        inp = full[:-1]
        lab = full[1:]
        forbidden = set(full)
        neg = [
            _sample_sasrec_negative(rng, forbidden)
            for _ in lab
        ]
        pairs.append((inp, lab, neg))

    if not pairs:
        return None

    lengths = torch.tensor(
        [len(inp) for inp, _, _ in pairs],
        dtype=torch.long,
        device=device,
    )
    max_len = int(lengths.max().item())

    input_tokens = torch.zeros(
        len(pairs),
        max_len,
        dtype=torch.long,
        device=device,
    )
    labels = torch.full(
        (len(pairs), max_len),
        IGNORE_INDEX,
        dtype=torch.long,
        device=device,
    )
    negative_labels = torch.full(
        (len(pairs), max_len),
        IGNORE_INDEX,
        dtype=torch.long,
        device=device,
    )

    for row, (inp, lab, neg) in enumerate(pairs):
        L = len(inp)

        input_tokens[row, :L] = torch.tensor(
            [item + 1 for item in inp],
            dtype=torch.long,
            device=device,
        )
        labels[row, :L] = torch.tensor(
            lab,
            dtype=torch.long,
            device=device,
        )
        negative_labels[row, :L] = torch.tensor(
            neg,
            dtype=torch.long,
            device=device,
        )

    return input_tokens, lengths, labels, negative_labels



# ===== 5. Fixed 1:49 evaluator =====


def deterministic_val_negatives(
    uid_str,
    sequence,
    target_item,
    num_negatives=49,
):
    # Exact Katcher LF validation protocol:
    # all catalog items are eligible except the target itself.
    pool = np.array(
        [item for item in range(NUM_ITEMS) if item != int(target_item)],
        dtype=np.int64,
    )
    uid_seed = (
        VAL_NEGATIVE_SEED
        + zlib.crc32(uid_str.encode("utf-8"))
    ) % (2**32)
    rng = np.random.default_rng(uid_seed)
    return rng.choice(
        pool,
        size=num_negatives,
        replace=False,
    ).tolist()


def single_sequence_to_tensor(sequence):
    if not sequence:
        return None
    input_tokens = torch.tensor(
        [[int(item) + 1 for item in sequence]],
        dtype=torch.long,
        device=device,
    )
    lengths = torch.tensor(
        [len(sequence)],
        dtype=torch.long,
        device=device,
    )
    return input_tokens, lengths


def evaluate_1to49(
    model,
    target_dict: Dict[str, int],
    sequence_dict: Dict[str, List[int]],
    negative_test_dict=None,
    k_list=(5, 10),
    num_negatives=49,
):
    model.eval()

    all_ndcg = {
        k: []
        for k in k_list
    }

    evaluated_users = 0

    with torch.no_grad():
        for uid_str, target_item in target_dict.items():
            if uid_str not in sequence_dict:
                continue

            sequence = sequence_dict[uid_str]
            prepared = single_sequence_to_tensor(
                sequence
            )

            if prepared is None:
                continue

            target_item = int(target_item)

            if not 0 <= target_item < NUM_ITEMS:
                continue

            if (
                negative_test_dict is not None
                and uid_str in negative_test_dict
            ):
                negatives = [
                    int(item)
                    for item in negative_test_dict[
                        uid_str
                    ]
                ][:num_negatives]
            else:
                negatives = deterministic_val_negatives(
                    uid_str=uid_str,
                    sequence=sequence,
                    target_item=target_item,
                    num_negatives=num_negatives,
                )

                if negatives is None:
                    continue

            if len(negatives) != num_negatives:
                continue

            candidates = [
                target_item,
                *negatives,
            ]

            candidate_tensor = torch.tensor(
                candidates,
                dtype=torch.long,
                device=device,
            )

            input_tokens, lengths = prepared

            candidate_logits = model.score_candidates(
                input_tokens=input_tokens,
                lengths=lengths,
                candidate_ids=candidate_tensor,
            )

            ranked_local = torch.argsort(
                candidate_logits,
                descending=True,
            ).tolist()

            # local candidate index 0이 target
            target_rank = ranked_local.index(0)

            for k in k_list:
                if target_rank < k:
                    ndcg = (
                        1.0
                        / np.log2(target_rank + 2)
                    )
                else:
                    ndcg = 0.0

                all_ndcg[k].append(
                    float(ndcg)
                )

            evaluated_users += 1

    print("=" * 70)
    print(
        "SASRec 평가 결과 "
        "(고정 1:49 candidate scoring)"
    )
    print("=" * 70)

    for k in k_list:
        if not all_ndcg[k]:
            raise RuntimeError(
                f"NDCG@{k} 평가 대상이 0명입니다."
            )

        print(
            f"NDCG@{k:2d}: "
            f"{np.mean(all_ndcg[k]):.4f} "
            f"(users={len(all_ndcg[k])})"
        )

    print("=" * 70)

    return all_ndcg, evaluated_users


# ===== Grouped fixed 1:49 TEST evaluator =====
def _finalize_grouped(vals):
    summary = {}
    for group in ("overall", "novel", "repeat"):
        summary[group] = {
            "n": len(vals[group][10]),
            "ndcg@5": float(np.mean(vals[group][5])) if vals[group][5] else None,
            "ndcg@10": float(np.mean(vals[group][10])) if vals[group][10] else None,
        }
    return summary


def evaluate_grouped_test(model):
    model.eval()
    vals = {
        g: {5: [], 10: []}
        for g in ("overall", "novel", "repeat")
    }
    per_user = {}

    with torch.no_grad():
        for uid, target_item in test_targets.items():
            sequence = test_sequences.get(uid, [])
            prepared = single_sequence_to_tensor(sequence)
            if prepared is None:
                continue

            target_item = int(target_item)
            negatives = [int(x) for x in negative_test_dict[uid]][:49]
            if len(negatives) != 49:
                continue

            candidates = [target_item] + negatives
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
            input_tokens, lengths = prepared

            scores = model.score_candidates(input_tokens, lengths, candidate_tensor)
            ranked_local = torch.argsort(scores, descending=True).tolist()
            rank = ranked_local.index(0)

            group = "repeat" if target_item in TEST_SEEN_ITEMS[uid] else "novel"
            user_rec = {"group": group}

            for k in (5, 10):
                ndcg = 1.0 / np.log2(rank + 2) if rank < k else 0.0
                vals["overall"][k].append(float(ndcg))
                vals[group][k].append(float(ndcg))
                user_rec[f"ndcg@{k}"] = float(ndcg)
            per_user[uid] = user_rec

    return _finalize_grouped(vals), per_user


def print_grouped(summary):
    for group in ("overall", "novel", "repeat"):
        row = summary[group]
        print(
            f"{group:7s} | "
            f"NDCG@5={row['ndcg@5']:.4f} | "
            f"NDCG@10={row['ndcg@10']:.4f} | "
            f"n={row['n']}"
        )


# ===== 5-seed training helper: ORIGINAL SASRec BCE objective =====
def train_one_seed(seed):
    global SEED, BEST_MODEL_PATH

    SEED = int(seed)
    set_seed(SEED)

    BEST_MODEL_PATH = os.path.join(
        CKPT_DIR,
        f"best_katcher_sasrec_original_bce1neg_seed{SEED}.pt",
    )

    model = SASRecBaseline(
        num_items=NUM_ITEMS,
        hidden_dim=HIDDEN_DIM,
        max_seq_len=MAX_ITEM_SEQ_LEN,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_ndcg10 = float("-inf")
    patience_counter = 0

    train_uid_list = list(train_users)

    print("학습 유저 수:", len(train_uid_list))
    print("model: PURE SASRec baseline")
    print("loss: ORIGINAL SASRec | BCE with 1 sampled negative per positive position")
    print("negative sampling: excludes items in the user's training sequence")
    print("evaluation: fixed 1 positive + 49 negatives (unchanged)")
    print("-" * 72)

    for epoch in range(EPOCHS):
        model.train()

        # Seed-specific deterministic shuffling and negative resampling.
        rng = np.random.default_rng(SEED * 1_000_003 + epoch)
        shuffled_users = list(train_uid_list)
        rng.shuffle(shuffled_users)

        total_loss = 0.0
        step_count = 0
        valid_predictions = 0

        for start in range(0, len(shuffled_users), BATCH_SIZE):
            batch_uids = shuffled_users[start:start + BATCH_SIZE]

            prepared = make_sequence_training_batch(
                batch_uids,
                train_sequences,
                train_targets,
                rng,
            )

            if prepared is None:
                continue

            input_tokens, lengths, labels, negative_labels = prepared

            hidden = model.encode_all(input_tokens)
            valid_mask = labels.ne(IGNORE_INDEX)

            if not bool(valid_mask.any()):
                continue

            hidden_valid = hidden[valid_mask]
            positive_ids = labels[valid_mask]
            negative_ids = negative_labels[valid_mask]

            positive_emb = model.item_embedding(positive_ids + 1)
            negative_emb = model.item_embedding(negative_ids + 1)

            positive_logits = (hidden_valid * positive_emb).sum(dim=-1)
            negative_logits = (hidden_valid * negative_emb).sum(dim=-1)

            # Numerically stable form of:
            # -log(sigmoid(pos)) - log(1 - sigmoid(neg))
            loss = (
                F.softplus(-positive_logits)
                + F.softplus(negative_logits)
            ).mean()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite SASRec BCE loss at epoch={epoch+1}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            optimizer.step()

            total_loss += float(loss.item())
            step_count += 1
            valid_predictions += int(valid_mask.sum().item())

        if step_count == 0:
            raise RuntimeError("학습 가능한 SASRec batch가 없습니다.")

        avg_loss = total_loss / step_count

        if (epoch + 1) % EVAL_EVERY == 0:
            print(
                f"Epoch {epoch+1:3d}/{EPOCHS} | "
                f"BCE loss={avg_loss:.4f} | "
                f"positive positions={valid_predictions}"
            )

            val_result, val_users_evaluated = evaluate_1to49(
                model,
                val_targets,
                val_sequences,
                negative_test_dict=None,
                k_list=(10,),
                num_negatives=49,
            )

            val_ndcg10 = float(np.mean(val_result[10]))

            print(
                f"Val NDCG@10={val_ndcg10:.4f} | "
                f"users={val_users_evaluated}"
            )

            if val_ndcg10 > best_val_ndcg10:
                best_val_ndcg10 = val_ndcg10
                patience_counter = 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print("  -> best 저장:", BEST_MODEL_PATH)
            else:
                patience_counter += 1
                print(f"  -> 미개선 {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if not os.path.exists(BEST_MODEL_PATH):
        raise RuntimeError("best SASRec checkpoint가 생성되지 않았습니다.")

    print(
        "학습 완료 | "
        f"best val NDCG@10={best_val_ndcg10:.6f}"
    )

    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=device)
    )
    model.eval()

    return model, float(best_val_ndcg10)


# ===== Run seeds 42..46 + Novel/Repeat summary =====
all_seed_results = []

for seed in SEEDS:
    print("\n" + "=" * 90)
    print(f"KATCHER | SASRec | SEED={seed}")
    print("=" * 90)

    model, best_val = train_one_seed(seed)
    grouped, _ = evaluate_grouped_test(model)

    print_grouped(grouped)

    all_seed_results.append({
        "seed": int(seed),
        "best_val_ndcg@10": float(best_val),
        "test": grouped,
    })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

aggregate = {}
for group in ("overall", "novel", "repeat"):
    aggregate[group] = {}
    for metric in ("ndcg@5", "ndcg@10"):
        arr = np.array(
            [r["test"][group][metric] for r in all_seed_results],
            dtype=np.float64,
        )
        aggregate[group][metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
        }
    aggregate[group]["n"] = int(all_seed_results[0]["test"][group]["n"])

final = {
    "dataset": "Katcher",
    "model": "SASRec",
    "training_objective": "original-style binary cross-entropy with 1 sampled negative per positive position",
    "training_negatives_per_positive": 1,
    "seeds": SEEDS,
    "protocol": "LF-aligned split/history/validation criterion/fixed 1:49 test; SASRec original-style BCE objective",
    "subset_definition": "target membership in FULL observed pre-target history",
    "per_seed": all_seed_results,
    "aggregate": aggregate,
}

summary_path = os.path.join(
    RESULT_ROOT,
    "katcher_sasrec_original_bce1neg_5seed_novel_repeat_summary.json",
)
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 90)
print("5-SEED FINAL SUMMARY")
print("=" * 90)
for group in ("overall", "novel", "repeat"):
    m5 = aggregate[group]["ndcg@5"]
    m10 = aggregate[group]["ndcg@10"]
    print(
        f"{group:7s} | "
        f"NDCG@5={m5['mean']:.4f} ± {m5['std']:.4f} | "
        f"NDCG@10={m10['mean']:.4f} ± {m10['std']:.4f} | "
        f"n={aggregate[group]['n']}"
    )
print("saved:", summary_path)

