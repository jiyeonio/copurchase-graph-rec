# Paths and run helpers of the run scripts (source this file). The same locations are defined for Python in paths.py.
# The defaults are folders of this repository; override the roots with KATCHERS_SPLIT, KATCHERS_DATA, INSTACART_DATA, RESULTS, PAPER and the interpreter with PY.
export CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PY=${PY:-python}
export KATCHERS_SPLIT=${KATCHERS_SPLIT:-$CODE/data/katchers_split}   # Katcher's train/val/test_data.csv: output of preprocess/katchers_split.py
export KATCHERS_DATA=${KATCHERS_DATA:-$CODE/data/katchers}       # Katcher's data: output of preprocess/katchers.py
export INSTACART_DATA=${INSTACART_DATA:-$CODE/data/instacart}    # Instacart data: output of preprocess/instacart.py, text and graph embeddings
export RESULTS=${RESULTS:-$CODE/results}                         # everything the scripts write
export PAPER=${PAPER:-$CODE/paper}                               # tables/, figs/, numbers.tex
KC=$KATCHERS_DATA; IC=$INSTACART_DATA; S=$RESULTS
RUNS=$S/runs; AN=$S/analysis; PU=$AN/pu; PU2=$AN/pu2; RK=$AN/rank       # runs, analysis files, per-user dumps, rank dumps
RUNS_EDGE=$S/runs_edge; PU_EDGE=$AN/pu_edge                            # removal of edges
RUNS_ENC=$S/runs_en; PU_ENC=$AN/pu_en                                  # English encoder on Instacart
GR=$S/graphs; IG=$S/ic_graphs; BF=$S/baselines_full
KC_META=$KC; KC_TEXT=$S/text/kc/item_embeddings_text.npy
IC_TEXT=$IC/item_embeddings_text.npy; IC_TEXT_EN=$S/text/ic_en/item_embeddings_text.npy
mkdir -p $RUNS $AN $PAPER/tables $PAPER/figs
SEEDS="42 43 44 45 46"
# Training runs. Defaults of the model scripts are the settings of the paper (d=128, dropout 0.3, Adam lr 2.5e-4, at most 100 epochs,
# validation every 2 (Katcher's) / 4 (Instacart) epochs, patience 5, training on every order with the final target weighted by 2).
kc_train(){ # name src graph_dir [extra flags]   -> $RUNS/kc11/<name>
  local n=$1 src=$2 gd=$3; shift 3
  $PY $CODE/model/model_katchers.py --exp kc11/$n --src $src --text-path $KC_TEXT --graph-dir $gd --out-root $RUNS --seeds $SEEDS --test "$@" > $AN/kc_$n.log 2>&1; echo "kc11/$n rc=$?"; }
ic_train(){ # name src graph_dir [extra flags]   -> $RUNS/ic11/<name>
  local n=$1 src=$2 gd=$3; shift 3
  $PY $CODE/model/model_instacart.py --exp ic11/$n --src $src --graph-dir $gd --out-root $RUNS --seeds $SEEDS --test "$@" > $AN/ic_$n.log 2>&1; echo "ic11/$n rc=$?"; }
dump(){ echo "--dump-pu $1_1to49_seed{seed}.json --dump-full $1_full_seed{seed}.json"; }   # per-user NDCG (1:49 and full catalogue), prefix as argument
