#!/bin/bash
# Instacart. The co-purchase graph (f>=3, $IC/graph/adj_freq3.npz) is written by preprocess/instacart.py and its LightGCN embeddings
# ($IC/item_embeddings_graph_freq3_seed{S}.npy) by: lightgcn_g.py --dataset instacart --data-dir $IC --freq 3.
# This script builds the rewired control: one rewired graph per seed and LightGCN on it -> $IG/rw_co_lg/emb. (GPU)
source "$(dirname "$0")/../config.sh"; cd $CODE/graphs
mkdir -p $IG/rw_co_lg/emb
for s in $SEEDS; do D=$IG/rw_co_$s; mkdir -p $D/graph; ln -sf $IC/item_meta.json $D/item_meta.json
  $PY rewire.py --dataset instacart --data-dir $IC --emb-dir $D/tmp --out-dir $D/rewire --seeds $s --swap-multiplier 3.0 --adj-only
  cp $D/rewire/adj_freq3_rewire_seed$s.npz $D/graph/adj_freq3.npz
  $PY lightgcn_g_fast.py --dataset instacart --data-dir $D --freq 3 --emb-dir $IG/rw_co_lg/emb --seeds $s --checkpoint-dir $D/ckpt_lg
done
