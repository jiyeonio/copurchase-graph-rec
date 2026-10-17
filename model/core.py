"""Transaction-level Transformer of the paper: model, evaluation and training loop shared by the two datasets.

Item representations: text = Linear(768->d)+LayerNorm+GELU of the fixed text embedding; graph = the same projection of the fixed
LightGCN embedding of the co-purchase graph; id = a learned embedding table (compared model).
Encoder: position embeddings per order (the items of one order share a position), block-causal attention (symmetric within an
order, causal across orders), pre-norm Transformer with 2 layers, 4 heads and feed-forward width 4d; the state of an order is the hidden
state of its last token.
Scoring and fusion: every representation scores the whole catalogue through an output layer tied to it; with text+graph the two
representations are read by the same Transformer and their scores are combined with learned softmax weights.
Training: next-order prediction at every order of the history with a full-softmax cross-entropy, the final target weighted
by 2; Adam, lr 2.5e-4, batch 64, at most 100 epochs, checkpoint selected by validation 1:49 NDCG@10 with patience 5.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRAPH_RAW_DIM, TEXT_RAW_DIM, MAX_ORDERS = 64, 768, 50
TF_LAYERS, TF_HEADS, TF_FF_MULT = 2, 4, 4
BATCH_SIZE, GRAD_CLIP, FINAL_W = 64, 5.0, 2.0
ITEM_POS_MAX = 1024

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def parse_args(data_dir, eval_every, val_batch, text_required):
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, help="run name; results go to <out-root>/<exp>/")
    p.add_argument("--src", required=True, choices=["text+graph", "text", "graph", "id"], help="item representation (text+graph = proposed model)")
    p.add_argument("--graph-dir", default=None, help="folder with the graph embedding of each seed (needed when --src contains graph)")
    p.add_argument("--text-path", default=None, required=text_required, help="item text embedding (.npy)")
    p.add_argument("--data-dir", default=data_dir)
    p.add_argument("--out-root", default="runs")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--eval-every", type=int, default=eval_every, help="validate every N epochs")
    p.add_argument("--patience", type=int, default=5, help="validations without improvement before stopping")
    p.add_argument("--val-batch", type=int, default=val_batch)
    p.add_argument("--test", action="store_true", help="evaluate the test users with the checkpoint selected on validation")
    p.add_argument("--eval-only", action="store_true", help="no training: load best_base_real_seed{S}.pt from --ckpt-dir and evaluate")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--last-order-only", action="store_true", help="variant: train on the final target only (Appendix, training rule)")
    p.add_argument("--item-level", action="store_true", help="variant: one position per item and an item-level causal mask (backbone variant)")
    p.add_argument("--no-compile", action="store_true", help="do not use torch.compile for the training loss")
    p.add_argument("--dump-pu", default=None, help="per-user test NDCG under 1:49 ranking, path with {seed}")
    p.add_argument("--dump-full", default=None, help="per-user test NDCG under full-catalogue ranking (history removed), path with {seed}")
    p.add_argument("--dump-rank", default=None, help="per-user test rank under full-catalogue ranking, path with {seed} and {mask}")
    p.add_argument("--dump-case", default=None, help="text+graph only: top-10 of each representation for every test user, path with {seed}")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_history(events, num_items, keep=lambda e: True, key=lambda e: (e["day"], e["order_idx"])):
    """events of one user -> (item ids, order position of each item); the last MAX_ORDERS orders, items of an order sorted by id."""
    item_ids, tx_pos, tx = [], [], 0
    for e in sorted(events, key=key)[-MAX_ORDERS:]:
        if not keep(e):
            continue
        items = sorted({int(i) for i in e.get("items", []) if 0 <= int(i) < num_items})
        if not items:
            continue
        item_ids.extend(items)
        tx_pos.extend([tx] * len(items))
        tx += 1
    if not item_ids:
        return None
    return torch.tensor(item_ids, dtype=torch.long), torch.tensor(tx_pos, dtype=torch.long)


def build_cache(histories, flatten):
    c = {}
    for uid, evs in histories.items():
        f = flatten(evs)
        if f is not None:
            c[str(uid)] = f
    return c


class Data:
    """What the training loop needs from a dataset: num_items, text (N x 768 tensor), train_cache / train_targets, and for val and test
    cache, targets ({uid: {"items": [target]}}), candidates ({uid: [target, 49 negatives]}) and seen ({uid: items of the history})."""
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def split(self, name):
        return (self.val_candidates, self.val_cache, self.val_targets, self.val_seen) if name == "val" else \
               (self.test_candidates, self.test_cache, self.test_targets, self.test_seen)


def collate_users(uids, cache, target_dict, num_items):
    seqs, txs, kept, targets = [], [], [], []
    for uid in uids:
        uid = str(uid)
        if uid not in cache or uid not in target_dict:
            continue
        ti = target_dict[uid].get("items", [])
        if not ti or not (0 <= int(ti[0]) < num_items):
            continue
        targets.append(int(ti[0]))
        s_, t_ = cache[uid]
        seqs.append(s_)
        txs.append(t_)
        kept.append(uid)
    if not seqs:
        return None
    B, ml = len(seqs), max(len(x) for x in seqs)
    ip = torch.zeros((B, ml), dtype=torch.long)
    tp = torch.zeros((B, ml), dtype=torch.long)
    vm = torch.zeros((B, ml), dtype=torch.bool)
    for r, (s_, t_) in enumerate(zip(seqs, txs)):
        ip[r, : len(s_)] = s_
        tp[r, : len(s_)] = t_
        vm[r, : len(s_)] = True
    return kept, ip.to(device), tp.to(device), vm.to(device), torch.tensor(targets, dtype=torch.long, device=device)


# ---------------------------------------------------------------- model
class Encoder(nn.Module):
    def __init__(self, hidden_dim, dropout, item_level):
        super().__init__()
        self.item_level = item_level
        self.transaction_position_embedding = nn.Embedding(MAX_ORDERS, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=TF_HEADS, dim_feedforward=hidden_dim * TF_FF_MULT,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=TF_LAYERS, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        if item_level:
            self.item_position_embedding = nn.Embedding(ITEM_POS_MAX, hidden_dim)

    def _mask(self, tx_pos, valid_mask, dtype):
        B, S = tx_pos.shape
        future = tx_pos[:, None, :] > tx_pos[:, :, None]                     # block-causal: an item attends to its own and earlier orders
        if self.item_level:
            ar = torch.arange(S, device=tx_pos.device)
            future = (ar[None, None, :] > ar[None, :, None]).expand(B, S, S)
        m = torch.zeros(B, S, S, dtype=dtype, device=tx_pos.device).masked_fill(future, float("-inf"))
        m = m[:, None].expand(B, TF_HEADS, S, S).reshape(B * TF_HEADS, S, S)
        kp = torch.zeros(B, S, dtype=dtype, device=tx_pos.device).masked_fill(~valid_mask, float("-inf"))
        return m, kp

    def forward(self, tokens, tx_pos, valid):
        """tokens: one (B,S,H) sequence per representation. All of them pass through the same Transformer (stacked along the batch)."""
        V, B = len(tokens), tokens[0].shape[0]
        safe = tx_pos.long().clamp(0, MAX_ORDERS - 1)
        pos = self.transaction_position_embedding(safe)
        if self.item_level:
            idx = torch.arange(tx_pos.shape[1], device=tx_pos.device).clamp(max=ITEM_POS_MAX - 1)
            pos = self.item_position_embedding(idx)[None].expand(B, -1, -1)
        x = torch.cat([t + pos for t in tokens], 0)
        am, kp = self._mask(torch.cat([safe] * V, 0), torch.cat([valid] * V, 0), x.dtype)
        h = self.output_norm(self.transformer(x, mask=am, src_key_padding_mask=kp))
        return [h[i * B:(i + 1) * B] for i in range(V)]


def last_order_state(h, tx_pos, valid):
    """hidden state of the last token of the last order"""
    B, S, H = h.shape
    pos = torch.arange(S, device=h.device).unsqueeze(0).expand(B, S)
    last_tx = torch.where(valid, tx_pos, torch.full_like(tx_pos, -1)).max(dim=1).values
    li = torch.where(valid & (tx_pos == last_tx[:, None]), pos, torch.full_like(pos, -1)).max(dim=1).values.clamp_min(0)
    return h[torch.arange(B, device=h.device), li]


def order_states(M, h):
    """states of all orders (B,K,H): hidden state of the last token of each order. M (B,K,S) marks the positions of every order."""
    B, K, S = M.shape
    pos = torch.arange(1, S + 1, device=M.device, dtype=torch.float32)
    li = ((M.float() * pos).amax(-1).long() - 1).clamp_min(0)
    return h[torch.arange(B, device=h.device)[:, None], li]


class TransactionModel(nn.Module):
    def __init__(self, src, num_items, hidden_dim, dropout, text_emb, graph_emb, item_level=False):
        super().__init__()
        d = hidden_dim
        self.num_items, self.views = num_items, src.split("+")
        if "text" in self.views: self.p_text = nn.Sequential(nn.Linear(TEXT_RAW_DIM, d), nn.LayerNorm(d), nn.GELU())
        if "graph" in self.views: self.p_graph = nn.Sequential(nn.Linear(GRAPH_RAW_DIM, d), nn.LayerNorm(d), nn.GELU())
        if "id" in self.views:
            self.id_emb = nn.Embedding(num_items, d); nn.init.normal_(self.id_emb.weight, mean=0.0, std=0.02)
        self.sequence_encoder = Encoder(d, dropout, item_level)
        if self.views == ["id"]:        # ID model: positions at the scale of the item table, output scores not divided by sqrt(d)
            nn.init.normal_(self.sequence_encoder.transaction_position_embedding.weight, mean=0.0, std=0.02)
            if item_level: nn.init.normal_(self.sequence_encoder.item_position_embedding.weight, mean=0.0, std=0.02)
        self.head_scale = 1.0 if self.views == ["id"] else 1.0 / math.sqrt(d)
        if len(self.views) > 1: self.fusion_logits = nn.Parameter(torch.zeros(len(self.views)))
        self.text_emb, self.graph_emb = text_emb, graph_emb

    def weights(self):
        if len(self.views) == 1: return {"x": torch.ones((), device=device)}
        w = torch.softmax(self.fusion_logits, 0)
        return {v: w[i] for i, v in enumerate(self.views)}

    def R(self, view, ids=None):
        """item vectors of one representation (used as input tokens and as output layer)"""
        if ids is None: ids = torch.arange(self.num_items, device=device)
        if view == "id": return self.id_emb(ids)
        return self.p_text(self.text_emb[ids]) if view == "text" else self.p_graph(self.graph_emb[ids])

    def hidden(self, item_ids, tx_pos, valid):
        return self.sequence_encoder([self.R(v, item_ids) for v in self.views], tx_pos, valid)

    def view_scores(self, states, candidate_ids=None):
        out = {}
        for v in self.views:
            u = states[v]
            s = (u @ self.R(v).T.to(u.dtype)) if candidate_ids is None else torch.einsum("bh,bch->bc", u, self.R(v, candidate_ids).to(u.dtype))
            out[v] = s
        return out

    def score(self, states, candidate_ids=None):
        s = self.view_scores(states, candidate_ids)
        if len(self.views) == 1: return s[self.views[0]] * self.head_scale
        w = self.weights(); total = 0
        for v in self.views: total = total + w[v].to(s[v].dtype) * s[v] * self.head_scale
        return total

    def user_states(self, item_ids, tx_pos, valid):
        H = self.hidden(item_ids, tx_pos, valid)
        return {v: last_order_state(H[i], tx_pos, valid) for i, v in enumerate(self.views)}

    def forward(self, item_ids, tx_pos, valid, candidate_ids=None):
        return self.score(self.user_states(item_ids, tx_pos, valid), candidate_ids)

    def prefix_states(self, item_ids, tx_pos, valid):
        H = self.hidden(item_ids, tx_pos, valid)
        n_tx = torch.where(valid, tx_pos, torch.full_like(tx_pos, -1)).max(dim=1).values + 1
        kk = torch.arange(MAX_ORDERS, device=H[0].device)
        M = (valid[:, None, :] & (tx_pos[:, None, :] == kk[None, :, None])).to(H[0].dtype)
        return {v: order_states(M, H[i]) for i, v in enumerate(self.views)}, M.sum(-1) > 0, n_tx

    def dense_loss(self, item_ids, tx_pos, valid, final_target, final_w):
        """state of order k predicts the items of order k+1 (one cross-entropy term per item); the state of the last order predicts the
        held-out target, weighted by final_w"""
        states, _, n_tx = self.prefix_states(item_ids, tx_pos, valid)
        B, K, N = item_ids.shape[0], MAX_ORDERS, self.num_items
        logits = self.score({v: x.reshape(B * K, -1) for v, x in states.items()}, None).view(B, K, N)
        logp = F.log_softmax(logits.float(), dim=-1)
        sel = valid & (tx_pos >= 1)
        bidx = torch.arange(B, device=logp.device)[:, None].expand_as(tx_pos)
        flat = (bidx * K + (tx_pos.long() - 1).clamp_min(0)) * N + item_ids.long()
        T = torch.zeros(B * K * N, device=logp.device).index_add_(0, flat.reshape(-1), sel.float().reshape(-1)).view(B, K, N)
        dense_sum = -(T * logp).sum(-1)
        final_logp = logp[torch.arange(B, device=logp.device), (n_tx - 1).clamp_min(0)]
        L_final = F.nll_loss(final_logp, final_target, reduction="none")
        return (dense_sum.sum() + final_w * L_final.sum()) / (T.sum() + final_w * B)


# ---------------------------------------------------------------- evaluation
@torch.inference_mode()
def eval_1to49(model, data, split, val_batch):
    """target against 49 fixed negatives -> per-user NDCG@10, NDCG@5, group (repeat: the target is in the history)"""
    model.eval()
    CAND, CACHE, TDATA, SEEN = data.split(split)
    uids = list(CAND.keys()); p10, p5, grp = {}, {}, {}
    for s in range(0, len(uids), val_batch):
        col = collate_users(uids[s:s + val_batch], CACHE, TDATA, data.num_items)
        if col is None:
            continue
        kept, ip, tp, vm, tg = col
        cand = torch.tensor([CAND[u] for u in kept], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            logits = model(ip, tp, vm, candidate_ids=cand)
        rank = (torch.argsort(logits.float(), dim=1, descending=True) == 0).nonzero(as_tuple=False)[:, 1]
        for r, u in enumerate(kept):
            rk = int(rank[r])
            p10[u] = 1.0 / np.log2(rk + 2) if rk < 10 else 0.0
            p5[u] = 1.0 / np.log2(rk + 2) if rk < 5 else 0.0
            grp[u] = "repeat" if int(tg[r]) in SEEN[u] else "novel"
    return p10, p5, grp


def summarize(per_user, group):
    v = np.array(list(per_user.values()))
    out = {"overall": float(v.mean()), "n": len(v)}
    for g in ("novel", "repeat"):
        x = np.array([per_user[u] for u in per_user if group[u] == g])
        out[g] = float(x.mean()) if len(x) else None
        out["n_" + g] = len(x)
    return out


def seen_mask(seen, num_items):
    m = torch.zeros(num_items, dtype=torch.bool, device=device)
    idx = torch.tensor(sorted(seen), dtype=torch.long, device=device)
    if len(idx): m[idx] = True
    return m


@torch.inference_mode()
def eval_full(model, data, split, masked, val_batch):
    """target against the whole catalogue. masked: the items of the user's history are removed (the target is never removed).
    Ties get the mid-rank. -> per-user NDCG@10, NDCG@5, group, 0-based rank"""
    model.eval()
    CAND, CACHE, TDATA, SEEN = data.split(split)
    uids = list(CAND.keys()); p10, p5, grp, rnk = {}, {}, {}, {}
    for s in range(0, len(uids), val_batch):
        col = collate_users(uids[s:s + val_batch], CACHE, TDATA, data.num_items)
        if col is None:
            continue
        kept, ip, tp, vm, tg = col
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            logits = model(ip, tp, vm)
        logits = logits.float(); B = logits.shape[0]; ar = torch.arange(B, device=logits.device)
        t_score = logits[ar, tg][:, None]
        if masked:
            m = torch.stack([seen_mask(SEEN[u], data.num_items) for u in kept]); m[ar, tg] = False
            logits = logits.masked_fill(m, float("-inf"))
        rank = ((logits > t_score).sum(1) + 0.5 * ((logits == t_score).sum(1) - 1)).float()
        n10 = torch.where(rank < 10, 1.0 / torch.log2(rank + 2), torch.zeros_like(rank)); n5 = torch.where(rank < 5, 1.0 / torch.log2(rank + 2), torch.zeros_like(rank))
        for r, u in enumerate(kept):
            p10[u] = float(n10[r]); p5[u] = float(n5[r]); grp[u] = "repeat" if int(tg[r]) in SEEN[u] else "novel"; rnk[u] = float(rank[r])
    return p10, p5, grp, rnk


@torch.inference_mode()
def dump_case(model, data, path, val_batch):
    """top-10 of the text scores, of the graph scores and of their combination for every test user (full catalogue, history removed)"""
    CAND, CACHE, TDATA, SEEN = data.split("test"); uids = list(CAND.keys()); case = {}
    for s in range(0, len(uids), val_batch):
        col = collate_users(uids[s:s + val_batch], CACHE, TDATA, data.num_items)
        if col is None:
            continue
        kept, ip, tp, vm, tg = col
        st = {v: x.float() for v, x in model.user_states(ip, tp, vm).items()}; w = model.weights()
        sc = {v: (st[v] @ model.R(v).float().T) * model.head_scale for v in model.views}
        sc["fused"] = sum(w[v].float() * sc[v] for v in model.views)
        for r, u in enumerate(kept):
            m = seen_mask(SEEN[u], data.num_items); m[tg[r]] = False
            e = {"target": int(tg[r]), "history": [int(x) for x in ip[r][vm[r]].tolist()], "tx": [int(x) for x in tp[r][vm[r]].tolist()]}
            for k, x in sc.items():
                y = x[r].masked_fill(m, float("-inf")); e[k + "_top"] = [int(i) for i in y.topk(10).indices.tolist()]; e[k + "_rank"] = int((y > y[tg[r]]).sum())
            case[u] = e
    json.dump({"weights": {k: float(v) for k, v in model.weights().items()}, "users": case}, open(path, "w"))


@torch.no_grad()
def assert_prefix_causal(model, data, n_users=64):
    """Leak check for the training targets: the state of order k must not change when the orders after k are replaced by random items."""
    model.eval()
    _, ip, tp, vm, _ = collate_users(list(data.train_cache.keys())[:n_users], data.train_cache, data.train_targets, data.num_items)
    n_tx = torch.where(vm, tp, torch.full_like(tp, -1)).max(1).values + 1
    cut = n_tx // 2
    s1, sv, _ = model.prefix_states(ip, tp, vm)
    g = torch.Generator(device="cpu").manual_seed(0)
    rnd = torch.randint(0, data.num_items, ip.shape, generator=g).to(ip.device)
    s2, _, _ = model.prefix_states(torch.where(vm & (tp > cut[:, None]), rnd, ip), tp, vm)
    kk = torch.arange(sv.shape[1], device=ip.device)[None, :]
    chk, later = (kk <= cut[:, None]) & sv, (kk > cut[:, None]) & sv
    assert bool(chk.any())
    for v in s1:
        d = (s1[v] - s2[v]).abs().amax(-1)
        assert d[chk].max().item() < 1e-4, f"order state depends on later orders ({v})"
        if bool(later.any()):
            assert d[later].max().item() > 1e-4, f"vacuous leak test ({v})"
    print(f"prefix-causal check OK for {list(s1)} ({int(chk.sum())} states)", flush=True)


# ---------------------------------------------------------------- training
def stage_train(data):
    """pad every training user once and keep the tensors on the GPU"""
    rows, items, txs, lens, tgs = {}, [], [], [], []
    for u in data.train_uids:
        u = str(u)
        if u not in data.train_cache or u not in data.train_targets or not data.train_targets[u].get("items"):
            continue
        t = int(data.train_targets[u]["items"][0])
        if not (0 <= t < data.num_items):
            continue
        s_, t_ = data.train_cache[u]
        rows[u] = len(lens); items.append(s_); txs.append(t_); lens.append(len(s_)); tgs.append(t)
    L = max(lens); U = len(lens)
    I = torch.zeros((U, L), dtype=torch.long); X = torch.zeros((U, L), dtype=torch.long)
    for r, (s_, t_) in enumerate(zip(items, txs)):
        I[r, :len(s_)] = s_; X[r, :len(s_)] = t_
    return rows, I.to(device), X.to(device), torch.tensor(lens, device=device), torch.tensor(tgs, dtype=torch.long, device=device)


def staged_batch(stage, batch_uids):
    rows, I, X, LEN, TG = stage
    r = [rows[str(u)] for u in batch_uids if str(u) in rows]
    if not r:
        return None
    r = torch.tensor(r, device=device); lens = LEN[r]; S = int(lens.max())
    valid = torch.arange(S, device=device)[None, :] < lens[:, None]
    return I[r, :S], X[r, :S], valid, TG[r]


def run(args, data, graph_file):
    """graph_file(graph_dir, seed) -> path of the graph embedding of that seed"""
    out = Path(args.out_root) / args.exp
    out.mkdir(parents=True, exist_ok=True)
    epochs = 0 if args.eval_only else args.epochs
    for seed in args.seeds:
        set_seed(seed)
        graph_emb = None
        if "graph" in args.src:
            graph_emb = torch.as_tensor(np.load(graph_file(args.graph_dir, seed)), dtype=torch.float32, device=device)
            assert tuple(graph_emb.shape) == (data.num_items, GRAPH_RAW_DIM) and torch.isfinite(graph_emb).all()
        model = TransactionModel(args.src, data.num_items, args.hidden, args.dropout, data.text, graph_emb, args.item_level).to(device)
        lossfn = model.dense_loss if args.no_compile else torch.compile(model.dense_loss, dynamic=True)
        print(f"MODEL src={args.src} seed={seed} item_level={args.item_level} last_order_only={args.last_order_only} "
              f"max_epochs={epochs} eval_every={args.eval_every} patience={args.patience} compile={not args.no_compile}", flush=True)
        if not args.last_order_only:
            assert_prefix_causal(model, data)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, fused=True)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(AMP_DTYPE == torch.float16))
        best, best_ep, pat, t0 = float("-inf"), None, 0, time.time()
        ckpt = (Path(args.ckpt_dir) if args.eval_only else out) / f"best_base_real_seed{seed}.pt"
        uids = list(data.train_uids)
        stage = stage_train(data)
        for epoch in range(epochs):
            model.train()
            np.random.shuffle(uids)
            tot, steps = torch.zeros((), device=device), 0
            for s in range(0, len(uids), BATCH_SIZE):
                col = staged_batch(stage, uids[s:s + BATCH_SIZE])
                if col is None:
                    continue
                ip, tp, vm, tg = col
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    if args.last_order_only:
                        loss = F.cross_entropy(model(ip, tp, vm), tg)
                    else:
                        loss = lossfn(ip, tp, vm, tg, FINAL_W)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                tot += loss.detach()
                steps += 1
            if (epoch + 1) % args.eval_every == 0:
                p10, _, gr = eval_1to49(model, data, "val", args.val_batch)
                v = summarize(p10, gr)["overall"]
                print(f"[{args.exp}] seed={seed} ep={epoch + 1} loss={float(tot) / steps:.4f} val@10={v:.4f} {(time.time() - t0) / 60:.1f}m", flush=True)
                if v > best:
                    best, best_ep, pat = v, epoch + 1, 0
                    torch.save(model.state_dict(), ckpt)
                else:
                    pat += 1
                    if pat >= args.patience:
                        break
        model.load_state_dict(torch.load(ckpt, map_location=device))
        p10, p5, gr = eval_1to49(model, data, "val", args.val_batch)
        res = {"seed": seed, "src": args.src, "best_epoch": best_ep, "val": summarize(p10, gr), "val_ndcg5": summarize(p5, gr),
               "view_weights": {k: float(v) for k, v in model.weights().items()},
               "config": {"hidden": args.hidden, "dropout": args.dropout, "lr": args.lr, "graph_dir": args.graph_dir, "text_path": args.text_path,
                          "item_level": args.item_level, "last_order_only": args.last_order_only, "eval_every": args.eval_every,
                          "patience": args.patience, "max_epochs": args.epochs},
               "elapsed_min": (time.time() - t0) / 60}
        if args.test:
            p10, p5, gr = eval_1to49(model, data, "test", args.val_batch)
            res["test"] = summarize(p10, gr); res["test_ndcg5"] = summarize(p5, gr)
            if args.dump_pu:
                multi = {u: bool(torch.bincount(tx).max() > 1) for u, (_, tx) in data.test_cache.items()}     # history has an order with several items
                json.dump({u: {"n10": float(p10[u]), "n5": float(p5[u]), "grp": gr[u], "multi": multi[u]} for u in p10}, open(args.dump_pu.replace("{seed}", str(seed)), "w"))
        for sp in (("val", "test") if args.test else ("val",)):
            for masked in (True, False):
                a, b, g, rk = eval_full(model, data, sp, masked, args.val_batch); tag = f"{sp}_full_" + ("masked" if masked else "unmasked")
                res[tag] = summarize(a, g); res[tag + "_ndcg5"] = summarize(b, g)
                if sp == "test" and masked and args.dump_full:
                    json.dump({u: {"n10": a[u], "n5": b[u], "grp": g[u]} for u in a}, open(args.dump_full.replace("{seed}", str(seed)), "w"))
                if sp == "test" and args.dump_rank:
                    json.dump({u: {"rank": rk[u], "grp": g[u]} for u in a}, open(args.dump_rank.replace("{seed}", str(seed)).replace("{mask}", "masked" if masked else "unmasked"), "w"))
        if args.test and args.dump_case and len(model.views) > 1:
            dump_case(model, data, args.dump_case.replace("{seed}", str(seed)), args.val_batch)
        json.dump(res, open(out / f"result_U_seed{seed}.json", "w"), indent=2)
        print(json.dumps(res, indent=1), flush=True)
