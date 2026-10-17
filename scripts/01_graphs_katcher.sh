#!/bin/bash
# Katcher's: co-purchase graphs, LightGCN embeddings, rewired control, graphs with edges removed. Outputs under $GR. (GPU)
source "$(dirname "$0")/../config.sh"; cd $CODE/graphs
$PY build_graph_katchers.py                      # $GR/g1 (f>=1, the graph of the paper), $GR/gf2, $GR/gf3 (Appendix: threshold)
$PY lightgcn_g.py --dataset katchers --data-dir $GR/g1 --emb-dir $GR/g1/emb --seeds $SEEDS --checkpoint-dir $GR/g1/ckpt
for f in 2 3; do g=$GR/gf$f
  $PY lightgcn_g_fast.py --dataset katchers --data-dir $g --emb-dir $g/emb --seeds $SEEDS --checkpoint-dir $g/ckpt > $g/lightgcn.log 2>&1
done
# control: one degree-preserving rewiring of g1 per seed and LightGCN on each rewired graph
$PY rewire.py --dataset katchers --data-dir $GR/g1 --emb-dir $GR/g1/emb --out-dir $GR/g1/rewire --seeds $SEEDS
mkdir -p $GR/g1_rwlg/emb     # the rewired embeddings under the file names the model reads
for s in $SEEDS; do ln -sfn $GR/g1/emb/item_embeddings_graph_rewire_seed$s.npy $GR/g1_rwlg/emb/item_embeddings_graph_seed$s.npy; done
# Appendix (removal of edges): 25 / 50 / 75% of the edges of g1 removed
$PY drop_edges.py
for d in 25 50 75; do g=$GR/gdrop$d
  $PY lightgcn_g_fast.py --dataset katchers --data-dir $g --emb-dir $g/emb --seeds $SEEDS --checkpoint-dir $g/ckpt > $g/lightgcn.log 2>&1
done
