"""Baseline of the paper (Section 5.2; Table 2, Tables A1 and A3): BERT4Rec (Cloze, full-softmax CE)
Dataset: Katcher's
Run: python baselines/original/kc_bert4rec.py --data-dir <DATA_DIR> --emb-dir <EMB_DIR> --out-dir <OUT_DIR> (the paper's numbers: scripts/09_baselines_full.sh)
"""
import argparse as _argparse
import os as _os


def _parse_args():
    p = _argparse.ArgumentParser(description=__doc__, formatter_class=_argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="directory with the dataset json/csv files")
    p.add_argument("--emb-dir", default=None, help="directory with the *.npy embeddings (default: --data-dir)")
    p.add_argument("--out-dir", default="results/katchers/baseline_bert4rec", help="results / checkpoints are written here")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="random seeds")
    return p.parse_args()


ARGS = _parse_args()
if ARGS.emb_dir is None:
    ARGS.emb_dir = ARGS.data_dir
_os.makedirs(ARGS.out_dir, exist_ok=True)
RESULT_ROOT = ARGS.out_dir
CKPT_DIR = _os.path.join(ARGS.out_dir, "checkpoints")
_os.makedirs(CKPT_DIR, exist_ok=True)



# ===== 0. Config — LF-matched comparison protocol =====
import os
import random
import json
import zlib
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEEDS = ARGS.seeds
SEED = SEEDS[0]
VAL_NEGATIVE_SEED = 42
DATA_DIR = ARGS.data_dir
EMB_DIR = ARGS.emb_dir
ITEM_EMB_TEXT_PATH = os.path.join(EMB_DIR, "item_embeddings_text.npy")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.json")
TRAIN_TARGETS_PATH = os.path.join(DATA_DIR, "train_targets.json")
VAL_HISTORY_PATH = os.path.join(DATA_DIR, "val_history.json")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val_data.json")
TEST_HISTORY_PATH = os.path.join(DATA_DIR, "test_history.json")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.json")
NEGATIVE_TEST_PATH = os.path.join(DATA_DIR, "negative_test.json")

MAX_ORDERS = 50

HIDDEN_DIM = 64
NUM_BLOCKS = 2
NUM_HEADS = 2
FF_DIM = 256
DROPOUT = 0.2
MASK_PROB = 0.20

BATCH_SIZE = 128
LR = 1e-4
WEIGHT_DECAY = 0.01
EPOCHS = 30

# Match Katcher LF
EVAL_EVERY = 2
PATIENCE = 4

BEST_MODEL_PATH = os.path.join(CKPT_DIR, f"best_katcher_bert4rec_lfmatched_seed{SEED}.pt")
RESULT_PATH = os.path.join(RESULT_ROOT, "katcher_bert4rec_lfmatched_5seed.json")

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
print("device:", device)
print("protocol: Katcher LF-matched split/history/eval + canonical BERT4Rec Cloze training")



# ===== 1. Katcher data loading — exact LF dependencies =====
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

required = [
    ITEM_EMB_TEXT_PATH,
    TRAIN_DATA_PATH,
    TRAIN_TARGETS_PATH,
    VAL_HISTORY_PATH,
    VAL_DATA_PATH,
    TEST_HISTORY_PATH,
    TEST_DATA_PATH,
    NEGATIVE_TEST_PATH,
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
            raise ValueError(f"empty target items: {v}")
        return int(items[0])
    if isinstance(v, list):
        if not v:
            raise ValueError("empty target list")
        return int(v[0])
    return int(v)

train_users = [str(uid) for uid in train_data]
val_users = [str(uid) for uid in val_history]
test_users = [str(uid) for uid in test_history]

train_targets = {u: _target_to_int(train_targets_raw[u]) for u in train_users}
val_targets = {u: _target_to_int(val_data_raw[u]) for u in val_users}
test_targets = {u: _target_to_int(test_data_raw[u]) for u in test_users}

def _normalize_negs(v):
    if isinstance(v, dict):
        v = v.get("negs", [])
    return [int(x) for x in v]

negative_test_dict = {u: _normalize_negs(negative_raw[u]) for u in test_users}

assert set(train_users).isdisjoint(val_users)
assert set(train_users).isdisjoint(test_users)
assert set(val_users).isdisjoint(test_users)

def _full_seen_from_events(events):
    seen = set()
    for ev in events:
        for x in ev.get("items", []):
            x = int(x)
            if 0 <= x < NUM_ITEMS:
                seen.add(x)
    return seen

TEST_SEEN_ITEMS = {u: _full_seen_from_events(test_history[u]) for u in test_users}

print("NUM_ITEMS:", NUM_ITEMS)
print("split train/val/test:", len(train_users), len(val_users), len(test_users))



# ===== 2. Exact Katcher LF history contract -> BERT4Rec item sequence =====
def sort_events(events):
    return sorted(
        events,
        key=lambda e: (e.get("day", 0), str(e.get("order_id", ""))),
    )

def build_observed_sequence_from_events(events):
    seq = []
    for ev in sort_events(events)[-MAX_ORDERS:]:
        valid_items = sorted({
            int(x)
            for x in ev.get("items", [])
            if 0 <= int(x) < NUM_ITEMS
        })
        seq.extend(valid_items)
    return seq

train_observed_sequences = {u: build_observed_sequence_from_events(train_data[u]) for u in train_users}
val_sequences = {u: build_observed_sequence_from_events(val_history[u]) for u in val_users}
test_sequences = {u: build_observed_sequence_from_events(test_history[u]) for u in test_users}

train_full_sequences = {
    u: train_observed_sequences[u] + [int(train_targets[u])]
    for u in train_users
}

MAX_HISTORY_ITEMS = max(
    max((len(v) for v in train_observed_sequences.values()), default=0),
    max((len(v) for v in val_sequences.values()), default=0),
    max((len(v) for v in test_sequences.values()), default=0),
)
MAX_SEQ_LEN = max(
    MAX_HISTORY_ITEMS + 1,
    max((len(v) for v in train_full_sequences.values()), default=1),
)

def sequence_stats(seq_dict):
    lengths = np.array([len(v) for v in seq_dict.values()], dtype=np.int64)
    return {
        "users": int(len(lengths)),
        "empty": int((lengths == 0).sum()),
        "mean": float(lengths.mean()) if len(lengths) else 0.0,
        "median": float(np.median(lengths)) if len(lengths) else 0.0,
        "max": int(lengths.max()) if len(lengths) else 0,
    }

print("train observed:", sequence_stats(train_observed_sequences))
print("train full:", sequence_stats(train_full_sequences))
print("val:", sequence_stats(val_sequences))
print("test:", sequence_stats(test_sequences))
print("dynamic MAX_SEQ_LEN:", MAX_SEQ_LEN)



# ===== 3. BERT4Rec-style model =====
# token IDs
#   PAD  = 0
#   item raw i -> i + 1
#   MASK = NUM_ITEMS + 1

PAD_TOKEN = 0
MASK_TOKEN = NUM_ITEMS + 1
VOCAB_SIZE = NUM_ITEMS + 2


class BERT4RecBaseline(nn.Module):
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

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            hidden_dim,
            padding_idx=PAD_TOKEN,
        )

        self.position_embedding = nn.Embedding(
            max_seq_len + 1,
            hidden_dim,
            padding_idx=0,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # BERT4Rec 핵심:
        # causal mask 없이 bidirectional self-attention
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_blocks,
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

        # BERT4Rec/BERT MLM-style prediction head
        # hidden -> dense -> GELU -> LayerNorm -> tied item embedding
        self.pred_dense = nn.Linear(hidden_dim, hidden_dim)
        self.pred_norm = nn.LayerNorm(hidden_dim)
        self.output_bias = nn.Parameter(torch.zeros(NUM_ITEMS))

        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=0.02,
        )
        nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        with torch.no_grad():
            self.token_embedding.weight[PAD_TOKEN].zero_()
            self.position_embedding.weight[0].zero_()

    def encode(
        self,
        input_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_tokens: [B, L]
        return: [B, L, H]
        """
        if input_tokens.ndim != 2:
            raise ValueError(
                "input_tokens는 [B, L] 텐서여야 합니다."
            )

        batch_size, seq_len = input_tokens.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} > max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(
            1,
            seq_len + 1,
            device=input_tokens.device,
        ).unsqueeze(0).expand(
            batch_size,
            seq_len,
        )

        positions = positions.masked_fill(
            input_tokens.eq(PAD_TOKEN),
            0,
        )

        x = (
            self.token_embedding(input_tokens)
            + self.position_embedding(positions)
        )
        x = self.embedding_dropout(x)

        padding_mask = input_tokens.eq(PAD_TOKEN)

        # IMPORTANT:
        # causal mask를 넣지 않는다 -> bidirectional attention
        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask,
        )

        x = self.output_norm(x)

        return x

    def item_logits_from_hidden(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden: [..., H]
        return: [..., NUM_ITEMS]

        BERT-style masked-item prediction head:
        Linear -> GELU -> LayerNorm -> tied item embedding + bias
        """
        h = self.pred_dense(hidden)
        h = F.gelu(h)
        h = self.pred_norm(h)

        item_embeddings = self.token_embedding.weight[1:NUM_ITEMS + 1]

        return torch.matmul(
            h,
            item_embeddings.T,
        ) + self.output_bias

    def score_candidates_at_mask(
        self,
        input_tokens: torch.Tensor,
        mask_position: int,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Validation/Test candidate scoring.

        IMPORTANT:
        Use exactly the same prediction head as training:
        hidden -> Linear -> GELU -> LayerNorm -> tied item embedding + item bias.
        This keeps training and evaluation score spaces identical.
        """
        hidden = self.encode(input_tokens)[0, mask_position]

        h = self.pred_dense(hidden)
        h = F.gelu(h)
        h = self.pred_norm(h)

        candidate_ids = candidate_ids.long()
        candidate_token_ids = candidate_ids + 1
        candidate_embeddings = self.token_embedding(candidate_token_ids)
        candidate_bias = self.output_bias[candidate_ids]

        return torch.matmul(candidate_embeddings, h) + candidate_bias



# ===== 4. Cloze masking + batch preparation =====

IGNORE_INDEX = -100


def bert4rec_mask_sequence(
    raw_sequence: List[int],
    rng: np.random.Generator,
    mask_prob: float = MASK_PROB,
):
    """
    BERT4Rec-style random Cloze masking.

    선택된 prediction position에 대해:
      - 80%: [MASK]
      - 10%: random item
      - 10%: original item 그대로

    labels에는 선택된 position의 원래 raw item ID를 저장하고,
    선택되지 않은 position은 IGNORE_INDEX.

    최소 1개 position은 반드시 prediction target으로 선택한다.
    """
    if not raw_sequence:
        return None

    seq = raw_sequence[-MAX_SEQ_LEN:]

    token_ids = [
        int(item) + 1
        for item in seq
    ]

    labels = [
        IGNORE_INDEX
        for _ in seq
    ]

    selected = (
        rng.random(len(seq))
        < mask_prob
    )

    # 최소 하나는 반드시 mask target
    if not selected.any():
        idx = int(
            rng.integers(
                low=0,
                high=len(seq),
            )
        )
        selected[idx] = True

    for pos in np.where(selected)[0]:
        original_raw_item = int(seq[pos])
        labels[pos] = original_raw_item

        p = float(rng.random())

        if p < 0.80:
            token_ids[pos] = MASK_TOKEN

        elif p < 0.90:
            random_raw_item = int(
                rng.integers(
                    low=0,
                    high=NUM_ITEMS,
                )
            )
            token_ids[pos] = random_raw_item + 1

        else:
            # 10% original token 그대로
            token_ids[pos] = original_raw_item + 1

    return token_ids, labels


def make_cloze_batch(
    user_ids: List[str],
    sequence_dict: Dict[str, List[int]],
    rng: np.random.Generator,
):
    masked_examples = []

    for uid in user_ids:
        seq = sequence_dict.get(uid, [])

        if not seq:
            continue

        result = bert4rec_mask_sequence(
            raw_sequence=seq,
            rng=rng,
            mask_prob=MASK_PROB,
        )

        if result is None:
            continue

        masked_examples.append(result)

    if not masked_examples:
        return None

    max_len = max(
        len(tokens)
        for tokens, _ in masked_examples
    )

    input_tokens = torch.full(
        (
            len(masked_examples),
            max_len,
        ),
        fill_value=PAD_TOKEN,
        dtype=torch.long,
        device=device,
    )

    labels = torch.full(
        (
            len(masked_examples),
            max_len,
        ),
        fill_value=IGNORE_INDEX,
        dtype=torch.long,
        device=device,
    )

    for row, (tokens, row_labels) in enumerate(
        masked_examples
    ):
        length = len(tokens)

        input_tokens[
            row,
            :length,
        ] = torch.tensor(
            tokens,
            dtype=torch.long,
            device=device,
        )

        labels[
            row,
            :length,
        ] = torch.tensor(
            row_labels,
            dtype=torch.long,
            device=device,
        )

    return input_tokens, labels


# quick check
rng_check = np.random.default_rng(SEED)

sample_batch = make_cloze_batch(
    user_ids=train_users[:4],
    sequence_dict=train_full_sequences,
    rng=rng_check,
)

if sample_batch is None:
    raise RuntimeError(
        "Cloze sample batch 생성 실패"
    )

sample_tokens, sample_labels = sample_batch

print("sample tokens shape:", tuple(sample_tokens.shape))
print("sample labels shape:", tuple(sample_labels.shape))
print(
    "masked prediction positions:",
    int(
        (sample_labels != IGNORE_INDEX)
        .sum()
        .item()
    ),
)



# ===== 5. Fixed 1:49 evaluator =====


def deterministic_val_negatives(
    uid_str: str,
    sequence: List[int],
    target_item: int,
    num_negatives: int = 49,
):
    # Exact Katcher LF validation rule:
    # sample from full catalog except target; fixed per-user seed.
    pool = np.array(
        [item for item in range(NUM_ITEMS) if item != int(target_item)],
        dtype=np.int64,
    )
    uid_seed = (
        VAL_NEGATIVE_SEED
        + zlib.crc32(uid_str.encode("utf-8"))
    ) % (2**32)
    rng = np.random.default_rng(uid_seed)
    return rng.choice(pool, size=num_negatives, replace=False).tolist()


def build_eval_input_with_mask(sequence: List[int]):
    if not sequence:
        return None

    token_ids = [int(item) + 1 for item in sequence]
    token_ids.append(MASK_TOKEN)
    input_tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    return input_tokens, len(token_ids) - 1


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

            prepared = build_eval_input_with_mask(
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

            input_tokens, mask_position = prepared

            candidate_logits = model.score_candidates_at_mask(
                input_tokens=input_tokens,
                mask_position=mask_position,
                candidate_ids=candidate_tensor,
            )

            ranked_local = torch.argsort(
                candidate_logits,
                descending=True,
            ).tolist()

            # local candidate index 0 = target
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
        "BERT4Rec 평가 결과 "
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
            prepared = build_eval_input_with_mask(sequence)
            if prepared is None:
                continue

            target_item = int(target_item)
            negatives = [int(x) for x in negative_test_dict[uid]][:49]
            if len(negatives) != 49:
                continue

            candidates = [target_item] + negatives
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
            input_tokens, mask_position = prepared

            scores = model.score_candidates_at_mask(
                input_tokens=input_tokens,
                mask_position=mask_position,
                candidate_ids=candidate_tensor,
            )
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


# ===== 5-seed training helper =====
def train_one_seed(seed):
    global SEED, BEST_MODEL_PATH

    SEED = int(seed)
    set_seed(SEED)
    BEST_MODEL_PATH = os.path.join(CKPT_DIR, f"best_katcher_bert4rec_seed{SEED}.pt")

    # ===== 6. Train: BERT4Rec-style Cloze objective =====

    model = BERT4RecBaseline(
        num_items=NUM_ITEMS,
        hidden_dim=HIDDEN_DIM,
        max_seq_len=MAX_SEQ_LEN,
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
    print("model: BERT4Rec (original-style bidirectional Cloze)")
    print(
        "loss: ORIGINAL BERT4Rec | Cloze masked-item Full-vocabulary Cross-Entropy"
    )
    print("bidirectional attention: True")
    print("temporal/text/graph: all False")
    print(f"training limit: max {EPOCHS} epochs | patience={PATIENCE}")
    print("-" * 72)

    for epoch in range(EPOCHS):
        model.train()

        # epoch별 deterministic masking/shuffle
        epoch_rng = np.random.default_rng(
            SEED + epoch
        )

        shuffled_users = list(train_uid_list)
        epoch_rng.shuffle(shuffled_users)

        total_loss = 0.0
        step_count = 0
        total_masked_positions = 0

        for start in range(
            0,
            len(shuffled_users),
            BATCH_SIZE,
        ):
            batch_uids = shuffled_users[
                start:start + BATCH_SIZE
            ]

            prepared = make_cloze_batch(
                user_ids=batch_uids,
                sequence_dict=train_full_sequences,
                rng=epoch_rng,
            )

            if prepared is None:
                continue

            input_tokens, labels = prepared

            hidden = model.encode(
                input_tokens
            )

            logits = model.item_logits_from_hidden(
                hidden
            )  # [B, L, NUM_ITEMS]

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    NUM_ITEMS,
                ),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss detected at epoch={epoch+1}"
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_loss += float(
                loss.item()
            )
            step_count += 1

            total_masked_positions += int(
                (labels != IGNORE_INDEX)
                .sum()
                .item()
            )

        if step_count == 0:
            raise RuntimeError(
                "학습 가능한 batch가 없습니다."
            )

        avg_loss = total_loss / step_count

        if (epoch + 1) % EVAL_EVERY == 0:
            print(
                f"Epoch {epoch + 1:3d}/{EPOCHS} | "
                f"loss={avg_loss:.4f} | "
                f"masked_positions={total_masked_positions}"
            )

            val_result, val_users_evaluated = evaluate_1to49(
                model=model,
                target_dict=val_targets,
                sequence_dict=val_sequences,
                negative_test_dict=None,
                k_list=(10,),
                num_negatives=49,
            )

            val_ndcg10 = float(
                np.mean(
                    val_result[10]
                )
            )

            print(
                f"Val NDCG@10={val_ndcg10:.4f} | "
                f"users={val_users_evaluated}"
            )

            if val_ndcg10 > best_val_ndcg10:
                best_val_ndcg10 = val_ndcg10
                patience_counter = 0

                torch.save(
                    model.state_dict(),
                    BEST_MODEL_PATH,
                )

                print(
                    "  -> best 저장:",
                    BEST_MODEL_PATH,
                )

            else:
                patience_counter += 1

                print(
                    f"  -> 미개선 "
                    f"{patience_counter}/{PATIENCE}"
                )

            if patience_counter >= PATIENCE:
                print(
                    f"Early stopping at epoch {epoch + 1}"
                )
                break

    print("-" * 72)
    print(
        "학습 완료 | "
        f"best val NDCG@10={best_val_ndcg10:.6f}"
    )

    # Reload the best validation checkpoint before test evaluation.
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=device)
    )
    model.eval()
    return model, float(best_val_ndcg10)


# ===== Run seeds 42..46 + Novel/Repeat summary =====
all_seed_results = []

for seed in SEEDS:
    print("\n" + "=" * 90)
    print(f"KATCHER | BERT4Rec | SEED={seed}")
    print("=" * 90)

    model, best_val = train_one_seed(seed)
    grouped, per_user = evaluate_grouped_test(model)

    print_grouped(grouped)

    per_user_path = os.path.join(
        RESULT_ROOT,
        f"katcher_bert4rec_seed{seed}_per_user.json",
    )
    with open(per_user_path, "w", encoding="utf-8") as f:
        json.dump(per_user, f, ensure_ascii=False, indent=2)

    all_seed_results.append({
        "seed": int(seed),
        "best_val_ndcg@10": float(best_val),
        "test": grouped,
        "per_user_path": per_user_path,
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
    "model": "BERT4Rec",
    "training_objective": "Cloze masked-item full-vocabulary softmax cross-entropy",
    "seeds": SEEDS,
    "protocol": "Katcher LF-matched split/history/fixed 1:49/best-val protocol; BERT4Rec canonical Cloze objective",
    "subset_definition": "target membership in FULL observed pre-target history",
    "per_seed": all_seed_results,
    "aggregate": aggregate,
}

summary_path = os.path.join(
    RESULT_ROOT,
    "katcher_bert4rec_original_objective_5seed_novel_repeat_summary.json",
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

