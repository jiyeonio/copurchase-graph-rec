"""Baseline of the paper (Section 5.2; Table 2, Tables A1 and A3): GRU4Rec (BPR-max)
Dataset: Instacart
Run: python baselines/original/ic_gru4rec.py --data-dir <DATA_DIR> --emb-dir <EMB_DIR> --out-dir <OUT_DIR> (the paper's numbers: scripts/09_baselines_full.sh)
"""
import argparse as _argparse
import os as _os


def _parse_args():
    p = _argparse.ArgumentParser(description=__doc__, formatter_class=_argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="directory with the dataset json/csv files")
    p.add_argument("--emb-dir", default=None, help="directory with the *.npy embeddings (default: --data-dir)")
    p.add_argument("--out-dir", default="results/instacart/baseline_gru4rec", help="results / checkpoints are written here")
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

SEEDS = ARGS.seeds
SEED = SEEDS[0]
DATA_DIR = ARGS.data_dir
EMB_DIR = ARGS.emb_dir
SPLIT_PATH = os.path.join(DATA_DIR, "split.json")
TARGET_PATH = os.path.join(DATA_DIR, "target.json")
NEGATIVE_TEST_PATH = os.path.join(DATA_DIR, "negative_test.json")
ITEM_META_PATH = os.path.join(DATA_DIR, "item_meta.json")
USER_ITEM_HISTORY_PATH = os.path.join(DATA_DIR, "user_item_history.json")

MAX_ORDERS = 50

EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_GRU_LAYERS = 1
DROPOUT = 0.0

BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100
EVAL_EVERY = 1
PATIENCE = 5

# BPR-max settings.
# Official GRU4Rec commonly uses ~1000–2000 extra samples; 1024 is a
# practical project-adapted default. These are validation-tunable.
BPR_N_SAMPLE = 1024
BPR_SAMPLE_ALPHA = 0.4
BPR_BPREG = 1.0
BPR_ELU_ALPHA = 0.5


VAL_NEGATIVE_SEED = 42
GRAPH_FREQ = 3

BEST_MODEL_PATH = os.path.join(CKPT_DIR, f"best_instacart_gru4rec_bprmax_adapted_seed{SEED}.pt")

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


# ===== 1. Load split / target / fixed negatives =====
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

for path in [SPLIT_PATH, TARGET_PATH, NEGATIVE_TEST_PATH, ITEM_META_PATH, USER_ITEM_HISTORY_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

split = load_json(SPLIT_PATH)
target = load_json(TARGET_PATH)
negative_raw = load_json(NEGATIVE_TEST_PATH)
item_meta = load_json(ITEM_META_PATH)
user_item_history = load_json(USER_ITEM_HISTORY_PATH)

train_users = [str(x) for x in split["train"]]
val_users = [str(x) for x in split["val"]]
test_users = [str(x) for x in split["test"]]

NUM_ITEMS = len(item_meta)

train_targets = {uid: int(target[uid]) for uid in train_users}
val_targets = {uid: int(target[uid]) for uid in val_users}
test_targets = {uid: int(target[uid]) for uid in test_users}
negative_test_dict = {
    uid: [int(x) for x in negative_raw[uid]["negs"]]
    for uid in test_users
}

assert set(train_users).isdisjoint(set(val_users))
assert set(train_users).isdisjoint(set(test_users))
assert set(val_users).isdisjoint(set(test_users))

bad = []
for uid in test_users:
    negs = negative_test_dict[uid][:49]
    t = test_targets[uid]
    if len(negs) != 49 or len(set(negs)) != 49 or t in negs:
        bad.append(uid)
if bad:
    raise ValueError(f"negative_test 정합성 오류 예시: {bad[:10]}")



# ===== 2. Build item-ID sequences =====
def build_item_sequence(uid: str):
    orders = user_item_history[uid][-MAX_ORDERS:]
    seq = []
    for order_items in orders:
        valid = sorted({
            int(item)
            for item in order_items
            if 0 <= int(item) < NUM_ITEMS
        })
        seq.extend(valid)
    return seq

train_sequences = {uid: build_item_sequence(uid) for uid in train_users}
val_sequences = {uid: build_item_sequence(uid) for uid in val_users}
test_sequences = {uid: build_item_sequence(uid) for uid in test_users}

MAX_ITEM_SEQ_LEN = max(
    max((len(v) for v in train_sequences.values()), default=1),
    max((len(v) for v in val_sequences.values()), default=1),
    max((len(v) for v in test_sequences.values()), default=1),
)


# ===== Novel / Repeat grouping (FULL observed history, not truncated model input) =====
# 논문 subset 정의를 model input truncation과 분리해 고정한다.
def _flatten_instacart_full_history(uid):
    items = []
    for order in user_item_history[str(uid)]:
        if isinstance(order, list):
            items.extend(int(x) for x in order if 0 <= int(x) < NUM_ITEMS)
    return items

TEST_SEEN_ITEMS = {
    str(uid): set(_flatten_instacart_full_history(uid))
    for uid in test_users
}


# ===== 3. PURE GRU4Rec baseline =====
# Architecture only: item embedding -> GRU -> linear item classifier.
# No attention, graph, memory, fusion, LayerNorm, residual block, or auxiliary encoder.

class GRU4RecBaseline(nn.Module):
    def __init__(
        self,
        num_items,
        embed_dim=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        self.num_items = int(num_items)

        self.item_embedding = nn.Embedding(
            num_items + 1,
            embed_dim,
            padding_idx=0,
        )
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, num_items)

        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode_all(self, input_tokens, lengths):
        if input_tokens.ndim != 2:
            raise ValueError("input_tokens must be [B, L].")
        if torch.any(lengths <= 0):
            raise ValueError("sequence length must be > 0")

        x = self.item_embedding(input_tokens)
        packed = pack_padded_sequence(
            x,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        from torch.nn.utils.rnn import pad_packed_sequence
        hidden, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=input_tokens.size(1),
        )
        return hidden

    def encode(self, input_tokens, lengths):
        hidden = self.encode_all(input_tokens, lengths)
        last_idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(input_tokens.size(0), device=input_tokens.device)
        return hidden[batch_idx, last_idx]

    def score_all_positions(self, input_tokens, lengths):
        return self.output(self.encode_all(input_tokens, lengths))

    def score_all_items(self, input_tokens, lengths):
        return self.output(self.encode(input_tokens, lengths))

    def score_candidates(self, input_tokens, lengths, candidate_ids):
        if input_tokens.size(0) != 1:
            raise ValueError("evaluation expects batch_size=1")
        logits = self.score_all_items(input_tokens, lengths)[0]
        return logits[candidate_ids]


# ===== 4. Sequence-level next-item batch preparation =====
# For each training user:
#   full = observed history + held-out training target
#   input  = full[:-1]
#   label  = full[1:]
# Thus every valid timestep contributes a next-item prediction loss.

IGNORE_INDEX = -100

def make_sequence_training_batch(
    user_ids,
    sequence_dict,
    target_dict,
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

        # Keep the FINAL held-out target while respecting model input length.
        full = (
            [int(x) for x in seq]
            + [tgt]
        )[-(MAX_ITEM_SEQ_LEN + 1):]

        if len(full) < 2:
            continue

        inp = full[:-1]
        lab = full[1:]

        pairs.append((inp, lab))

    if not pairs:
        return None

    lengths = torch.tensor(
        [len(inp) for inp, _ in pairs],
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

    for row, (inp, lab) in enumerate(pairs):
        L = len(inp)

        input_tokens[row, :L] = torch.tensor(
            [item + 1 for item in inp],
            dtype=torch.long,
            device=device,
        )

        # labels remain RAW item ids 0..NUM_ITEMS-1
        labels[row, :L] = torch.tensor(
            lab,
            dtype=torch.long,
            device=device,
        )

    return input_tokens, lengths, labels



# ===== 5. Fixed 1:49 evaluator =====
def deterministic_val_negatives(uid_str, sequence, target_item, num_negatives=49):
    seen = set(int(x) for x in sequence)
    pool = np.array(
        sorted(set(range(NUM_ITEMS)) - seen - {int(target_item)}),
        dtype=np.int64,
    )

    if len(pool) < num_negatives:
        return None

    uid_seed = (
        VAL_NEGATIVE_SEED
        + zlib.crc32(uid_str.encode("utf-8"))
        + GRAPH_FREQ * 1000
    ) % (2**32)

    rng = np.random.default_rng(uid_seed)
    return rng.choice(pool, size=num_negatives, replace=False).tolist()

def single_sequence_to_tensor(sequence):
    if not sequence:
        return None

    input_tokens = torch.tensor(
        [[int(item) + 1 for item in sequence]],
        dtype=torch.long,
        device=device,
    )
    lengths = torch.tensor([len(sequence)], dtype=torch.long, device=device)
    return input_tokens, lengths

def evaluate_1to49(model, target_dict, sequence_dict, negative_test_dict=None, k_list=(5, 10), num_negatives=49):
    model.eval()
    all_ndcg = {k: [] for k in k_list}
    evaluated_users = 0

    with torch.no_grad():
        for uid, target_item in target_dict.items():
            seq = sequence_dict.get(uid, [])
            prepared = single_sequence_to_tensor(seq)
            if prepared is None:
                continue

            if negative_test_dict is not None:
                negatives = [int(x) for x in negative_test_dict[uid]][:num_negatives]
            else:
                negatives = deterministic_val_negatives(uid, seq, target_item, num_negatives)
                if negatives is None:
                    continue

            if len(negatives) != num_negatives:
                continue

            candidates = [int(target_item)] + negatives
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)

            input_tokens, lengths = prepared
            scores = model.score_candidates(input_tokens, lengths, candidate_tensor)

            ranked_local = torch.argsort(scores, descending=True).tolist()
            target_rank = ranked_local.index(0)

            for k in k_list:
                ndcg = 1.0 / np.log2(target_rank + 2) if target_rank < k else 0.0
                all_ndcg[k].append(float(ndcg))

            evaluated_users += 1

    print("=" * 70)
    print("GRU4Rec-style 평가 결과 (고정 1:49 candidate scoring)")
    print("=" * 70)
    for k in k_list:
        print(f"NDCG@{k:2d}: {np.mean(all_ndcg[k]):.4f} (users={len(all_ndcg[k])})")
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


# ===== 5-seed training helper: GRU4Rec BPR-max objective =====
# This is a project-adapted BPR-max implementation:
# - same GRU architecture / project sequence construction / fixed 1:49 evaluation
# - other active rows' positives at the same timestep act as negatives
# - plus popularity^alpha sampled extra negatives
# - BPR-max weighting follows the GRU4Rec formulation
# - ELU(alpha=0.5) is used as the final activation for the BPR-max loss

def _build_bpr_sampling_probs():
    counts = np.zeros(NUM_ITEMS, dtype=np.float64)

    for uid in train_users:
        for item in train_sequences.get(uid, []):
            item = int(item)
            if 0 <= item < NUM_ITEMS:
                counts[item] += 1.0

        tgt = train_targets.get(uid)
        if tgt is not None:
            tgt = int(tgt)
            if 0 <= tgt < NUM_ITEMS:
                counts[tgt] += 1.0

    if counts.sum() <= 0:
        return np.full(NUM_ITEMS, 1.0 / NUM_ITEMS, dtype=np.float64)

    # GRU4Rec-style support^sample_alpha sampling.
    weights = np.power(counts, BPR_SAMPLE_ALPHA)

    # If some catalog items never occur in training, they simply receive zero
    # sampling probability, matching support-based negative sampling.
    if weights.sum() <= 0:
        return np.full(NUM_ITEMS, 1.0 / NUM_ITEMS, dtype=np.float64)

    return weights / weights.sum()


BPR_SAMPLING_PROBS = _build_bpr_sampling_probs()


def _sample_extra_negative_ids(rng):
    if NUM_ITEMS <= 1:
        raise RuntimeError("BPR-max requires at least two catalog items.")

    n_sample = int(min(BPR_N_SAMPLE, max(1, NUM_ITEMS - 1)))

    sampled = rng.choice(
        NUM_ITEMS,
        size=n_sample,
        replace=True,
        p=BPR_SAMPLING_PROBS,
    )

    # Removing duplicates reduces wasted columns without changing the sampling source.
    sampled = np.unique(sampled.astype(np.int64))

    if sampled.size == 0:
        sampled = np.array([0], dtype=np.int64)

    return torch.as_tensor(
        sampled,
        dtype=torch.long,
        device=device,
    )


def bpr_max_sequence_loss(
    model,
    input_tokens,
    lengths,
    labels,
    extra_negative_ids,
):
    """
    Project-adapted GRU4Rec BPR-max loss.

    At each timestep:
      1) active rows' true next items are placed first in the candidate set,
      2) other active rows' targets + sampled items serve as negatives,
      3) every candidate equal to the row's own positive item is masked out,
      4) negatives are softmax-weighted (BPR-max),
      5) the GRU4Rec score regularizer is added.

    Returns
    -------
    loss : scalar tensor
    n_events : int
    """
    hidden = model.encode_all(input_tokens, lengths)  # [B, L, H]

    total_loss_sum = hidden.new_zeros(())
    total_events = 0

    for t in range(labels.size(1)):
        active_mask = labels[:, t] != IGNORE_INDEX
        n_active = int(active_mask.sum().item())

        if n_active == 0:
            continue

        h_t = hidden[active_mask, t, :]               # [M, H]
        pos_ids = labels[active_mask, t].long()      # [M]

        # Candidate layout:
        #   first M columns = each active row's positive target
        #   remaining columns = sampled negatives
        candidate_ids = torch.cat(
            [pos_ids, extra_negative_ids],
            dim=0,
        )

        cand_w = model.output.weight[candidate_ids]   # [C, H]
        cand_b = (
            model.output.bias[candidate_ids]
            if model.output.bias is not None
            else None
        )

        # [M, C]
        scores = F.linear(h_t, cand_w, cand_b)

        # Recommended type of final activation for BPR-max.
        scores = F.elu(scores, alpha=BPR_ELU_ALPHA)

        row_idx = torch.arange(n_active, device=scores.device)
        pos_scores = scores[row_idx, row_idx]         # [M]

        # Exclude *all* columns containing the row's own positive item,
        # not only the diagonal; duplicates can occur across active rows
        # or in the sampled negatives.
        negative_mask = (
            candidate_ids.unsqueeze(0)
            != pos_ids.unsqueeze(1)
        )                                             # [M, C]

        valid_rows = negative_mask.any(dim=1)
        if not torch.all(valid_rows):
            scores = scores[valid_rows]
            pos_scores = pos_scores[valid_rows]
            negative_mask = negative_mask[valid_rows]

        if scores.size(0) == 0:
            continue

        # Softmax over negatives only.
        masked_neg_scores = scores.masked_fill(
            ~negative_mask,
            -1e9,
        )
        softmax_weights = F.softmax(
            masked_neg_scores,
            dim=1,
        )

        pair_prob = torch.sigmoid(
            pos_scores.unsqueeze(1) - scores
        )

        weighted_pair_prob = (
            pair_prob * softmax_weights
        ).sum(dim=1).clamp_min(1e-24)

        # Positive/masked columns must contribute zero to score regularization.
        reg_scores = torch.where(
            negative_mask,
            scores,
            torch.zeros_like(scores),
        )

        loss_rows = (
            -torch.log(weighted_pair_prob)
            + BPR_BPREG
            * (
                reg_scores.pow(2)
                * softmax_weights
            ).sum(dim=1)
        )

        total_loss_sum = total_loss_sum + loss_rows.sum()
        total_events += int(loss_rows.numel())

    if total_events == 0:
        raise RuntimeError(
            "BPR-max loss received no valid next-item events."
        )

    return total_loss_sum / total_events, total_events


def train_one_seed(seed):
    global SEED, BEST_MODEL_PATH

    SEED = int(seed)
    set_seed(SEED)

    BEST_MODEL_PATH = os.path.join(
        CKPT_DIR,
        f"best_instacart_gru4rec_bprmax_adapted_seed{SEED}.pt",
    )

    model = GRU4RecBaseline(
        NUM_ITEMS,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GRU_LAYERS,
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
    print("model: GRU4Rec architecture baseline (project-adapted sequence implementation)")
    print(
        "loss: BPR-max | "
        f"extra negatives={BPR_N_SAMPLE} | "
        f"sample_alpha={BPR_SAMPLE_ALPHA} | "
        f"bpreg={BPR_BPREG} | "
        f"ELU alpha={BPR_ELU_ALPHA}"
    )
    print(
        "negative source: other active targets at the same timestep "
        "+ support^alpha sampled extra negatives"
    )
    print("-" * 72)

    for epoch in range(EPOCHS):
        model.train()

        rng = np.random.default_rng(SEED + epoch)
        shuffled_users = list(train_uid_list)
        rng.shuffle(shuffled_users)

        total_loss = 0.0
        step_count = 0
        valid_predictions = 0

        for start in range(
            0,
            len(shuffled_users),
            BATCH_SIZE,
        ):
            batch_uids = shuffled_users[
                start:start + BATCH_SIZE
            ]

            prepared = make_sequence_training_batch(
                batch_uids,
                train_sequences,
                train_targets,
            )

            if prepared is None:
                continue

            input_tokens, lengths, labels = prepared

            extra_negative_ids = _sample_extra_negative_ids(rng)

            loss, n_events = bpr_max_sequence_loss(
                model,
                input_tokens,
                lengths,
                labels,
                extra_negative_ids,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite BPR-max loss at epoch={epoch+1}"
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
            valid_predictions += int(n_events)

        if step_count == 0:
            raise RuntimeError(
                "학습 가능한 batch가 없습니다."
            )

        avg_loss = total_loss / step_count

        if (epoch + 1) % EVAL_EVERY == 0:
            print(
                f"Epoch {epoch+1:3d}/{EPOCHS} | "
                f"BPR-max loss={avg_loss:.4f} | "
                f"next-item events={valid_predictions}"
            )

            val_result, val_users_evaluated = evaluate_1to49(
                model,
                val_targets,
                val_sequences,
                negative_test_dict=None,
                k_list=(10,),
                num_negatives=49,
            )

            val_ndcg10 = float(
                np.mean(val_result[10])
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
                    f"Early stopping at epoch {epoch+1}"
                )
                break

    print(
        "학습 완료 | "
        f"best val NDCG@10={best_val_ndcg10:.6f}"
    )

    model.load_state_dict(
        torch.load(
            BEST_MODEL_PATH,
            map_location=device,
        )
    )
    model.eval()

    return model, float(best_val_ndcg10)

# ===== Run seeds 42..46 + Novel/Repeat summary =====
all_seed_results = []

for seed in SEEDS:
    print("\n" + "=" * 90)
    print(f"INSTACART | GRU4Rec | SEED={seed}")
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
    "dataset": "Instacart",
    "model": "GRU4Rec",
    "seeds": SEEDS,
    "training_objective": "BPR-max with active-target and sampled negatives",
    "objective_hparams": {
        "n_sample": BPR_N_SAMPLE,
        "sample_alpha": BPR_SAMPLE_ALPHA,
        "bpreg": BPR_BPREG,
        "elu_alpha": BPR_ELU_ALPHA,
    },
    "implementation_note": "GRU4Rec architecture adapted to project user-history sequences; not an exact session-parallel reproduction",
    "protocol": "LF-aligned split/history/fixed 1:49 evaluation; validation NDCG@10 checkpoint selection",
    "subset_definition": "target membership in FULL observed pre-target history",
    "per_seed": all_seed_results,
    "aggregate": aggregate,
}

summary_path = os.path.join(
    RESULT_ROOT,
    "instacart_gru4rec_bprmax_adapted_5seed_novel_repeat_summary.json",
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
