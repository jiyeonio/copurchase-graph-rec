"""Graph representation of the items: LightGCN on the co-purchase graph, one embedding per seed.
64 dimensions, 3 propagation layers, mean of the layer outputs, BPR loss over the edges, Adam lr 1e-3, batch 1024, at most 200 epochs,
early stopping on the training loss with patience 20.
Katcher's : python graphs/lightgcn_g.py --dataset katchers --data-dir <graph folder with graph_edges.csv> --emb-dir <out> --checkpoint-dir <dir>
Instacart : python graphs/lightgcn_g.py --dataset instacart --data-dir <folder with graph/adj_freq3.npz> --freq 3 --emb-dir <out> --checkpoint-dir <dir>
Writes <emb-dir>/item_embeddings_graph[_freq3]_seed{S}.npy.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import N_ITEMS
from torch_geometric.nn import LGConv


LIGHTGCN_DIM = 64
LIGHTGCN_LAYERS = 3
LIGHTGCN_LR = 1e-3
LIGHTGCN_BATCH_SIZE = 1024
LIGHTGCN_EPOCHS = 200
LIGHTGCN_PATIENCE = 20


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class LightGCN(nn.Module):
    def __init__(self, num_items, embed_dim=64, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

    def forward(self, edge_index, edge_weight=None):
        x = self.embedding.weight
        layers = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            layers.append(x)
        return torch.stack(layers, dim=0).mean(dim=0)


def bpr_loss(embeddings, pos_pairs, neg_items):
    i_emb = embeddings[pos_pairs[:, 0]]
    j_emb = embeddings[pos_pairs[:, 1]]
    k_emb = embeddings[neg_items]

    pos_score = (i_emb * j_emb).sum(dim=1)
    neg_score = (i_emb * k_emb).sum(dim=1)

    return -F.logsigmoid(pos_score - neg_score).mean()


def build_katchers_graph(data_dir: Path):
    """Katcher's: read <data_dir>/graph_edges.csv (written by build_graph_katchers.py or drop_edges.py)."""
    import pandas as pd
    df = pd.read_csv(data_dir / "graph_edges.csv")
    num_items = N_ITEMS["kc"]
    src = df["item_i"].to_numpy(np.int64); dst = df["item_j"].to_numpy(np.int64); counts = df["count"].to_numpy(np.int64)
    assert (src < dst).all() and src.max() < num_items
    degree = np.bincount(np.concatenate([src, dst]), minlength=num_items)
    stats = {"num_items": num_items, "edges": int(len(src)), "isolated_items": int((degree == 0).sum()),
             "isolated_item_ratio": float((degree == 0).mean()), "max_count": int(counts.max())}
    print("graph variant:", stats)
    return num_items, src, dst, counts, stats, data_dir / "graph_edges.csv"


def load_graph(data_dir: Path, freq: int, dataset: str = "instacart"):
    if dataset == "katchers":
        num_items, src, dst, counts, stats, _ = build_katchers_graph(data_dir)
        counts32 = counts.astype(np.float32)
        weight = (counts32 / counts32.max()).astype(np.float32)   # normalised by the maximum retained frequency
        return num_items, src, dst, weight, stats

    meta_path = data_dir / "item_meta.json"
    adj_path = data_dir / "graph" / f"adj_freq{freq}.npz"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")
    if not adj_path.exists():
        raise FileNotFoundError(
            f"Missing: {adj_path}"
        )

    with meta_path.open("r", encoding="utf-8") as f:
        num_items = len(json.load(f))

    adj = sp.load_npz(adj_path)

    if adj.shape != (num_items, num_items):
        raise ValueError(
            f"adjacency shape mismatch: {adj.shape} vs {(num_items, num_items)}"
        )

    coo = adj.tocoo()

    # each undirected edge is used once as a positive pair
    keep = coo.row < coo.col
    src = coo.row[keep].astype(np.int64)
    dst = coo.col[keep].astype(np.int64)
    counts = coo.data[keep].astype(np.float64)

    if len(src) == 0:
        raise ValueError(f"No co-purchase edges remain for freq={freq}")

    # edge weight = co-purchase count / maximum count
    weight = (counts / counts.max()).astype(np.float32)

    stats = {
        "num_items": int(num_items),
        "freq": int(freq),
        "edges": int(len(src)),
        "max_count": float(counts.max()),
    }

    return num_items, src, dst, weight, stats


def train_one_seed(
    *,
    seed,
    freq,
    num_items,
    edge_index,
    edge_weight,
    pos_pairs,
    device,
    data_dir,
    checkpoint_dir,
    dataset="instacart",
    emb_dir=None,
):
    set_seed(seed)

    model = LightGCN(
        num_items=num_items,
        embed_dim=LIGHTGCN_DIM,
        num_layers=LIGHTGCN_LAYERS,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LIGHTGCN_LR,
    )

    tag = f"freq{freq}_" if dataset == "instacart" else ""
    emb_dir = data_dir if emb_dir is None else emb_dir
    checkpoint = (
        checkpoint_dir
        / f"lightgcn_best_{tag}seed{seed}.pt"
    )

    best_loss = float("inf")
    patience = 0
    n_pairs = len(pos_pairs)
    last_epoch = 0

    print(
        f"\n{'=' * 72}\n"
        f"{dataset} LightGCN | {tag}seed={seed}\n"
        f"{'=' * 72}",
        flush=True,
    )

    for epoch in range(LIGHTGCN_EPOCHS):
        last_epoch = epoch + 1
        model.train()

        perm = torch.randperm(n_pairs)

        total = 0.0
        n_batches = 0

        for start in range(0, n_pairs, LIGHTGCN_BATCH_SIZE):
            idx = perm[start:start + LIGHTGCN_BATCH_SIZE]
            pos_batch = pos_pairs[idx].to(device)

            # one negative item drawn uniformly from all items
            neg_batch = torch.randint(
                0,
                num_items,
                (len(idx),),
                device=device,
            )

            embeddings = model(edge_index, edge_weight)
            loss = bpr_loss(
                embeddings,
                pos_batch,
                neg_batch,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            n_batches += 1

        avg = total / max(n_batches, 1)

        if avg < best_loss:
            best_loss = avg
            patience = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience += 1

        if (epoch + 1) % 20 == 0:
            print(
                f"seed={seed} epoch={epoch+1} "
                f"loss={avg:.6f} best={best_loss:.6f} "
                f"patience={patience}/{LIGHTGCN_PATIENCE}",
                flush=True,
            )

        if patience >= LIGHTGCN_PATIENCE:
            print(
                f"early stop | seed={seed} epoch={epoch+1}",
                flush=True,
            )
            break

    model.load_state_dict(
        torch.load(checkpoint, map_location=device)
    )
    model.eval()

    with torch.no_grad():
        item_embeddings = (
            model(edge_index, edge_weight)
            .cpu()
            .numpy()
        )

    output_path = (
        emb_dir
        / f"item_embeddings_graph_{tag}seed{seed}.npy"
    )
    np.save(output_path, item_embeddings)

    print(
        f"saved {item_embeddings.shape} -> {output_path}",
        flush=True,
    )

    return {
        "seed": int(seed),
        "best_train_loss": float(best_loss),
        "epochs_run": int(last_epoch),
        "checkpoint": str(checkpoint),
        "embedding_path": str(output_path),
        "embedding_shape": list(item_embeddings.shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--dataset",
        choices=["instacart", "katchers"],
        default="instacart",
    )
    parser.add_argument(
        "--emb-dir",
        type=Path,
        default=None,
        help="where the embeddings are written (default: --data-dir)",
    )
    parser.add_argument(
        "--freq",
        type=int,
        choices=[1, 3, 5],
        default=3,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints_lightgcn_multiseed"),
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()

    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = data_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    emb_dir = (args.emb_dir or data_dir).resolve()
    emb_dir.mkdir(parents=True, exist_ok=True)

    num_items, src, dst, weight, stats = load_graph(
        data_dir,
        args.freq,
        args.dataset,
    )

    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)
    weight_t = torch.tensor(weight, dtype=torch.float32)

    edge_index = torch.stack(
        [
            torch.cat([src_t, dst_t]),
            torch.cat([dst_t, src_t]),
        ],
        dim=0,
    )
    edge_weight = torch.cat([weight_t, weight_t])
    pos_pairs = torch.stack([src_t, dst_t], dim=1)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    print(
        f"device={device} | num_items={num_items} | "
        f"freq={args.freq} | edges={len(pos_pairs)} | seeds={args.seeds}",
        flush=True,
    )

    results = []

    for seed in args.seeds:
        results.append(
            train_one_seed(
                seed=seed,
                freq=args.freq,
                num_items=num_items,
                edge_index=edge_index,
                edge_weight=edge_weight,
                pos_pairs=pos_pairs,
                device=device,
                data_dir=data_dir,
                checkpoint_dir=checkpoint_dir,
                dataset=args.dataset,
                emb_dir=emb_dir,
            )
        )

    summary = {
        "dataset": args.dataset,
        "graph_stats": stats,
        "seeds": args.seeds,
        "lightgcn_config": {
            "dim": LIGHTGCN_DIM,
            "layers": LIGHTGCN_LAYERS,
            "lr": LIGHTGCN_LR,
            "batch_size": LIGHTGCN_BATCH_SIZE,
            "epochs": LIGHTGCN_EPOCHS,
            "patience": LIGHTGCN_PATIENCE,
        },
        "results": results,
    }

    summary_path = (
        checkpoint_dir
        / (f"instacart_lightgcn_freq{args.freq}_multiseed_summary.json" if args.dataset == "instacart"
           else "katchers_lightgcn_multiseed_summary.json")
    )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nAll requested seeds finished. Summary -> {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
