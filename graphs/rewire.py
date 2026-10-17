"""Degree-preserving rewiring of the co-purchase graph (graph control). Every item keeps its number of edges and the multiset of edge
weights is kept; only what an item is connected to changes. One rewired graph per seed; unless --adj-only is given, LightGCN is then
trained on the rewired graph with the settings of the real graph.
Katcher's : python graphs/rewire.py --dataset katchers --data-dir <graph folder with graph_edges.csv> --emb-dir <out> --out-dir <out>
Instacart : python graphs/rewire.py --dataset instacart --data-dir <data folder with graph/adj_freq3.npz> --out-dir <out> --adj-only
Writes <out-dir>/adj_[freq3_]rewire_seed{S}.npz and <emb-dir>/item_embeddings_graph_[freq3_]rewire_seed{S}.npy.
"""
import argparse as _argparse
import os as _os


def _parse_args():
    p = _argparse.ArgumentParser(description=__doc__, formatter_class=_argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=".", help="directory with the dataset json/csv files")
    p.add_argument("--emb-dir", default=None, help="directory with the *.npy embeddings (default: --data-dir)")
    p.add_argument("--out-dir", default="results/instacart/rewire", help="results / checkpoints are written here")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="random seeds")
    p.add_argument("--swap-multiplier", default=3.0, help='Instacart: attempted double-edge swaps per edge', type=float)
    p.add_argument("--force", default=False, help='overwrite an existing embedding', action='store_true')
    p.add_argument("--swaps-per-edge", type=int, default=10, help="Katcher's: successful swaps per edge")
    p.add_argument("--dataset", default="instacart", choices=["instacart", "katchers"], help="graph to rewire")
    p.add_argument("--adj-only", action="store_true", help="write the rewired adjacency only (no LightGCN)")
    return p.parse_args()


ARGS = _parse_args()
if ARGS.emb_dir is None:
    ARGS.emb_dir = ARGS.data_dir
_os.makedirs(ARGS.out_dir, exist_ok=True)
# file-name tag: Instacart keeps its freq3 names, Katcher's has no freq in its names
_TAG = 'freq3_' if ARGS.dataset == 'instacart' else ''


import json
import os
import random
import sys
from pathlib import Path
import networkx as nx
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
LIGHTGCN_LR = 0.001
LIGHTGCN_BATCH_SIZE = 1024
LIGHTGCN_EPOCHS = 200
LIGHTGCN_PATIENCE = 20

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def find_real_adj(data_dir: Path):
    candidates = [data_dir / 'adj_freq3.npz', data_dir / 'graph' / 'adj_freq3.npz']
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError('adj_freq3.npz not found. Tried:\n- ' + '\n- '.join((str(p) for p in candidates)))

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


def load_real_graph(data_dir: Path):
    if ARGS.dataset == 'katchers':
        num_items, src, dst, counts, _stats, graph_path = build_katchers_graph(data_dir)
        return (num_items, src, dst, counts.astype(np.float64), graph_path)
    meta_path = data_dir / 'item_meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    with meta_path.open('r', encoding='utf-8') as f:
        num_items = len(json.load(f))
    adj_path = find_real_adj(data_dir)
    adj = sp.load_npz(adj_path).tocsr()
    if adj.shape != (num_items, num_items):
        raise ValueError(f'Adjacency shape={adj.shape}, expected={(num_items, num_items)}')
    coo = adj.tocoo()
    keep = coo.row < coo.col
    src = coo.row[keep].astype(np.int64)
    dst = coo.col[keep].astype(np.int64)
    counts = coo.data[keep].astype(np.float64)
    if len(src) == 0:
        raise RuntimeError('No undirected edges found.')
    return (num_items, src, dst, counts, adj_path)

def node_degree(num_items, src, dst):
    degree = np.zeros(num_items, dtype=np.int64)
    np.add.at(degree, src, 1)
    np.add.at(degree, dst, 1)
    return degree

def canonical_edges(src, dst):
    return {(int(min(a, b)), int(max(a, b))) for a, b in zip(src, dst)}

def rewire_degree_preserving(*, num_items, src, dst, counts, seed, swap_multiplier):
    """
    Degree-preserving topology randomization.

    The simple undirected degree of EVERY node is preserved exactly.
    The original edge-weight multiset is preserved exactly, but weights
    are randomly reassigned to the rewired topology.
    """
    set_seed(seed)
    # Self-loops (Katcher's only; Instacart has none, so that path is unchanged) stay FIXED with their
    # weights. Only the non-loop edges are rewired in a degree-preserving way.
    full_src = np.asarray(src, dtype=np.int64)
    full_dst = np.asarray(dst, dtype=np.int64)
    full_counts = np.asarray(counts, dtype=np.float64)
    loop_mask = full_src == full_dst
    n_loops = int(loop_mask.sum())
    loop_src, loop_dst, loop_counts = full_src[loop_mask], full_dst[loop_mask], full_counts[loop_mask]
    src, dst, counts = full_src[~loop_mask], full_dst[~loop_mask], full_counts[~loop_mask]
    graph = nx.Graph()
    graph.add_nodes_from(range(num_items))
    graph.add_edges_from(((int(a), int(b)) for a, b in zip(src, dst)))
    n_edges = graph.number_of_edges()
    if n_edges != len(src):
        raise RuntimeError('Unexpected duplicate/self edge in the real simple graph.')
    nswap = max(1, int(round(swap_multiplier * n_edges)))
    max_tries = max(nswap * 20, nswap + 1000)
    print(f'rewire seed={seed} | edges={n_edges} | nswap={nswap} | max_tries={max_tries}', flush=True)
    nx.double_edge_swap(graph, nswap=nswap, max_tries=max_tries, seed=seed)
    rewired_edges = np.asarray(list(graph.edges()), dtype=np.int64)
    rsrc = np.minimum(rewired_edges[:, 0], rewired_edges[:, 1])
    rdst = np.maximum(rewired_edges[:, 0], rewired_edges[:, 1])
    order = np.lexsort((rdst, rsrc))
    rsrc = rsrc[order]
    rdst = rdst[order]
    original_degree = node_degree(num_items, src, dst)
    rewired_degree = node_degree(num_items, rsrc, rdst)
    if not np.array_equal(original_degree, rewired_degree):
        raise AssertionError('Degree sequence was NOT preserved node-by-node.')
    rng = np.random.default_rng(seed)
    shuffled_counts = np.asarray(counts, dtype=np.float64)[rng.permutation(len(counts))]
    if not np.array_equal(np.sort(shuffled_counts), np.sort(counts)):
        raise AssertionError('Weight multiset changed.')
    real_set = canonical_edges(src, dst)
    rewire_set = canonical_edges(rsrc, rdst)
    overlap = len(real_set & rewire_set)
    overlap_ratio = overlap / len(real_set)
    stats = {'seed': int(seed), 'num_items': int(num_items), 'num_edges': int(len(rsrc)), 'swap_multiplier': float(swap_multiplier), 'requested_swaps': int(nswap), 'degree_preserved_exactly': True, 'weight_multiset_preserved_exactly': True, 'real_edge_overlap_count': int(overlap), 'real_edge_overlap_ratio': float(overlap_ratio), 'isolated_nodes': int(np.sum(rewired_degree == 0)), 'max_degree': int(rewired_degree.max()), 'mean_degree': float(rewired_degree.mean())}
    if n_loops:
        rsrc = np.concatenate([rsrc, loop_src])
        rdst = np.concatenate([rdst, loop_dst])
        shuffled_counts = np.concatenate([shuffled_counts, loop_counts])
    # Final checks on the FULL graph, self-loops included (a self-loop adds 2 to its node's degree).
    if len(rsrc) != len(full_src):
        raise AssertionError('Edge count changed.')
    full_original_degree = node_degree(num_items, full_src, full_dst)
    full_rewired_degree = node_degree(num_items, rsrc, rdst)
    if not np.array_equal(full_original_degree, full_rewired_degree):
        raise AssertionError('Degree sequence (self-loops included) was NOT preserved node-by-node.')
    if not np.array_equal(np.sort(shuffled_counts), np.sort(full_counts)):
        raise AssertionError('Weight multiset (self-loops included) changed.')
    if not np.array_equal(np.sort(rsrc[rsrc == rdst]), np.sort(full_src[full_src == full_dst])):
        raise AssertionError('Self-loops changed.')
    stats.update({'num_edges': int(len(rsrc)), 'self_loops_fixed': n_loops, 'num_edges_rewired': int(len(src)),
                  'edge_count_preserved_exactly': True, 'degree_preserved_including_self_loops': True,
                  'isolated_nodes': int(np.sum(full_rewired_degree == 0)), 'max_degree': int(full_rewired_degree.max()),
                  'mean_degree': float(full_rewired_degree.mean())})
    return (rsrc, rdst, shuffled_counts, stats)

def degree_preserving_rewire_katchers(src, dst, weights, counts, num_items, rewire_seed,
                                      swaps_per_edge=10, max_attempts_factor=100):
    """Katcher's: 2-switch on the undirected simple graph; counts / weights are shuffled over the rewired edges afterwards
    (multisets preserved)."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.int64)
    if np.any(src >= dst):
        raise ValueError("input graph must be canonical with src < dst (and have no self-loops)")
    edges = list(zip(src.tolist(), dst.tolist()))
    edge_set = set(edges)
    if len(edge_set) != len(edges):
        raise ValueError("input graph contains duplicate edges")
    m = len(edges)
    if m < 2:
        raise ValueError("not enough edges to rewire")
    rng = np.random.default_rng(int(rewire_seed))
    target_swaps = int(max(1, swaps_per_edge * m))
    max_attempts = int(max(target_swaps + 1000, max_attempts_factor * m))
    successful = 0
    attempts = 0
    while successful < target_swaps and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, m))
        j = int(rng.integers(0, m - 1))
        if j >= i:
            j += 1
        old1 = edges[i]
        old2 = edges[j]
        a, b = old1
        c, d = old2
        if rng.random() < 0.5:
            a, b = b, a
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4:
            continue
        new1 = (min(a, d), max(a, d))
        new2 = (min(c, b), max(c, b))
        if new1 == new2:
            continue
        if new1 in edge_set or new2 in edge_set:
            continue
        edge_set.remove(old1)
        edge_set.remove(old2)
        edge_set.add(new1)
        edge_set.add(new2)
        edges[i] = new1
        edges[j] = new2
        successful += 1
    if successful < target_swaps:
        print("[WARN]", f"successful swaps {successful:,}/{target_swaps:,}", f"after {attempts:,} attempts")
    rew_src = np.fromiter((e[0] for e in edges), dtype=np.int64, count=m)
    rew_dst = np.fromiter((e[1] for e in edges), dtype=np.int64, count=m)
    perm = rng.permutation(m)
    rew_counts = counts[perm].copy()
    rew_weights = weights[perm].copy()
    real_degree = node_degree(num_items, src, dst)
    rew_degree = node_degree(num_items, rew_src, rew_dst)
    if not np.array_equal(real_degree, rew_degree):
        raise AssertionError("FAILED: exact degree sequence not preserved")
    rew_edge_set = set(zip(rew_src.tolist(), rew_dst.tolist()))
    if len(rew_edge_set) != m:
        raise AssertionError("FAILED: duplicate rewired edges")
    if np.any(rew_src == rew_dst):
        raise AssertionError("FAILED: self-loop in rewired graph")
    if not np.array_equal(np.sort(counts), np.sort(rew_counts)):
        raise AssertionError("FAILED: co-purchase-count multiset not preserved")
    if not np.array_equal(np.sort(weights.astype(np.float64)), np.sort(rew_weights.astype(np.float64))):
        raise AssertionError("FAILED: edge-weight multiset not preserved")
    overlap_count = len(set(zip(src.tolist(), dst.tolist())) & rew_edge_set)
    qc = {"rewire_seed": int(rewire_seed), "num_nodes": int(num_items), "num_edges": int(m),
          "degree_preserved_exactly": True, "weight_multiset_preserved_exactly": True, "self_loops_fixed": 0,
          "target_successful_swaps": int(target_swaps), "successful_swaps": int(successful), "attempts": int(attempts),
          "real_edge_overlap_count": int(overlap_count), "real_edge_overlap_ratio": float(overlap_count / m),
          "isolated_nodes": int((real_degree == 0).sum()), "max_degree": int(real_degree.max())}
    return rew_src, rew_dst, rew_counts, rew_weights, qc

def save_rewired_adj(*, data_dir, seed, num_items, src, dst, counts):
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.concatenate([counts, counts]).astype(np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_items, num_items), dtype=np.float32).tocsr()
    path = Path(ARGS.out_dir) / f'adj_{_TAG}rewire_seed{seed}.npz'
    sp.save_npz(path, adj)
    return path

def build_graph_tensors(*, src, dst, counts, device):
    weight = (counts / counts.max()).astype(np.float32)
    src_t = torch.as_tensor(src, dtype=torch.long)
    dst_t = torch.as_tensor(dst, dtype=torch.long)
    weight_t = torch.as_tensor(weight, dtype=torch.float32)
    edge_index = torch.stack([torch.cat([src_t, dst_t]), torch.cat([dst_t, src_t])], dim=0).to(device)
    edge_weight = torch.cat([weight_t, weight_t]).to(device)
    pos_pairs = torch.stack([src_t, dst_t], dim=1)
    return (edge_index, edge_weight, pos_pairs)

def train_lightgcn(*, data_dir, checkpoint_dir, seed, num_items, src, dst, counts, device):
    set_seed(seed)
    edge_index, edge_weight, pos_pairs = build_graph_tensors(src=src, dst=dst, counts=counts, device=device)
    model = LightGCN(num_items=num_items, embed_dim=LIGHTGCN_DIM, num_layers=LIGHTGCN_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LIGHTGCN_LR)
    checkpoint = checkpoint_dir / f'lightgcn_rewire_{_TAG}seed{seed}.pt'
    best_loss = float('inf')
    patience = 0
    n_pairs = len(pos_pairs)
    last_epoch = 0
    for epoch in range(LIGHTGCN_EPOCHS):
        last_epoch = epoch + 1
        model.train()
        perm = torch.randperm(n_pairs)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_pairs, LIGHTGCN_BATCH_SIZE):
            idx = perm[start:start + LIGHTGCN_BATCH_SIZE]
            pos_batch = pos_pairs[idx].to(device)
            neg_batch = torch.randint(0, num_items, (len(idx),), device=device)
            embeddings = model(edge_index, edge_weight)
            loss = bpr_loss(embeddings, pos_batch, neg_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience += 1
        if last_epoch % 20 == 0:
            print(f'rewire {ARGS.dataset} {_TAG}seed={seed} epoch={last_epoch} loss={avg_loss:.6f} best={best_loss:.6f} patience={patience}/{LIGHTGCN_PATIENCE}', flush=True)
        if patience >= LIGHTGCN_PATIENCE:
            print(f'early stop | seed={seed} | epoch={last_epoch}', flush=True)
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        item_embeddings = model(edge_index, edge_weight).cpu().numpy()
    output_path = Path(ARGS.emb_dir) / f'item_embeddings_graph_{_TAG}rewire_seed{seed}.npy'
    np.save(output_path, item_embeddings)
    return {'seed': int(seed), 'embedding_path': str(output_path), 'embedding_shape': list(item_embeddings.shape), 'best_train_loss': float(best_loss), 'epochs_run': int(last_epoch), 'checkpoint': str(checkpoint)}


REWIRE_DATA_DIR = Path(ARGS.data_dir).resolve()
REWIRE_CHECKPOINT_DIR = (Path(ARGS.out_dir).resolve() / 'checkpoints_rewire')
REWIRE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REWIRE_SEEDS = ARGS.seeds
REWIRE_SWAP_MULTIPLIER = ARGS.swap_multiplier
REWIRE_FORCE = ARGS.force
rewire_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rewire_num_items, rewire_real_src, rewire_real_dst, rewire_real_counts, rewire_real_adj_path = load_real_graph(REWIRE_DATA_DIR)
rewire_real_degree = node_degree(rewire_num_items, rewire_real_src, rewire_real_dst)
print('=' * 88)
print(f"{ARGS.dataset.upper()} {_TAG.upper().rstrip('_')} — DEGREE-PRESERVING REWIRE + LIGHTGCN")
print('=' * 88)
print('data dir:', REWIRE_DATA_DIR)
print('real graph source:', rewire_real_adj_path)
print('device:', rewire_device)
print('items:', rewire_num_items)
print('real edges:', len(rewire_real_src))
print('real isolated:', int(np.sum(rewire_real_degree == 0)))
print('real self-loops:', int(np.sum(rewire_real_src == rewire_real_dst)))
print('seeds:', REWIRE_SEEDS)
print('swap multiplier:', REWIRE_SWAP_MULTIPLIER)
rewire_generation_results = {}
for seed in REWIRE_SEEDS:
    output_path = Path(ARGS.emb_dir) / f'item_embeddings_graph_{_TAG}rewire_seed{seed}.npy'
    if output_path.exists() and (not REWIRE_FORCE):
        print(f'[skip] seed={seed}: embedding already exists -> {output_path}')
        continue
    print('\n' + '=' * 88)
    print(f'REWIRE + LIGHTGCN | seed={seed}')
    print('=' * 88)
    if ARGS.dataset == 'katchers':
        # Katcher's: 2-switch with RNG seed 100000 + seed
        _w = rewire_real_counts.astype(np.float32)
        _w = _w / _w.max()
        rsrc, rdst, rcounts, _rweights, rewire_stats = degree_preserving_rewire_katchers(
            rewire_real_src, rewire_real_dst, _w, rewire_real_counts.astype(np.int64), rewire_num_items,
            rewire_seed=100000 + int(seed), swaps_per_edge=ARGS.swaps_per_edge, max_attempts_factor=100)
        rcounts = rcounts.astype(np.float64)
    else:
        rsrc, rdst, rcounts, rewire_stats = rewire_degree_preserving(num_items=rewire_num_items, src=rewire_real_src, dst=rewire_real_dst, counts=rewire_real_counts, seed=seed, swap_multiplier=REWIRE_SWAP_MULTIPLIER)
    adj_path = save_rewired_adj(data_dir=REWIRE_DATA_DIR, seed=seed, num_items=rewire_num_items, src=rsrc, dst=rdst, counts=rcounts)
    print('self-loops fixed:', rewire_stats['self_loops_fixed'], '| edges:', rewire_stats['num_edges'], '| degree preserved:', rewire_stats['degree_preserved_exactly'], '| weight multiset preserved:', rewire_stats['weight_multiset_preserved_exactly'], '| real-edge overlap:', f"{rewire_stats['real_edge_overlap_ratio']:.4%}")
    if ARGS.adj_only:
        train_stats = {'embedding_path': 'skipped (--adj-only)'}
    else:
        train_stats = train_lightgcn(data_dir=REWIRE_DATA_DIR, checkpoint_dir=REWIRE_CHECKPOINT_DIR, seed=seed, num_items=rewire_num_items, src=rsrc, dst=rdst, counts=rcounts, device=rewire_device)
    rewire_generation_results[str(seed)] = {'rewire': rewire_stats, 'lightgcn': train_stats, 'rewired_adj_path': str(adj_path)}
    with (REWIRE_CHECKPOINT_DIR / f'rewire_{_TAG}seed{seed}_summary.json').open('w', encoding='utf-8') as f:
        json.dump(rewire_generation_results[str(seed)], f, ensure_ascii=False, indent=2)
    print('embedding saved:', train_stats['embedding_path'])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
if not ARGS.adj_only:
    for seed in REWIRE_SEEDS:
        p = Path(ARGS.emb_dir) / f'item_embeddings_graph_{_TAG}rewire_seed{seed}.npy'
        if not p.exists():
            raise FileNotFoundError(p)
print('\nrewired graphs written for seeds', REWIRE_SEEDS)
