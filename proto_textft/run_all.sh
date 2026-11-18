#!/bin/bash
# Appendix H: partial fine-tuning of KR-ELECTRA: the last 2 of its 12 layers are trained end-to-end with the recommender (layers 1-10 frozen, their
# output cached once), [CLS] read; Text only and Text+Graph. Encoder settings: --enc-lr 2e-5 --enc-dropout 0; everything else is
# the default of model/core.py (the paper: batch 64, lr 2.5e-4, at most 100 epochs, validation 1:49 NDCG@10 every 2 (Katcher's) /
# 4 (Instacart) epochs, patience 5, gradient clipping 5.0); the training loss is not compiled.
#   bash proto_textft/run_all.sh cache     caches of the frozen layers (GPU, under a minute each)
#   bash proto_textft/run_all.sh train     training of every dataset and seed, test 1:49 and full catalogue, per-user dumps (GPU)
#   bash proto_textft/run_all.sh an        paired tests against the fixed encoder of the paper (CPU; needs the paper dumps in <PU>)
#   DATASETS=ic SEEDS=42 bash proto_textft/run_all.sh train      one dataset, one seed
# Environment: PY; KATCHERS_DATA, INSTACART_DATA, RESULTS as paths.py; KC_TEXT = Katcher's text embedding of the paper (KR-ELECTRA,
# regenerated); KC_GRAPH = folder of Katcher's graph embeddings of the paper; KC_TEXT_DATA = folder with Katcher's item_metadata.csv;
# CACHE = folder of the caches. The Instacart text and graph embeddings are read from INSTACART_DATA, as the paper's runs do.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"; PY=${PY:-python}
DATASETS=${DATASETS:-kc ic}; SEEDS=${SEEDS:-42 43 44 45 46}; TAG=$(echo $SEEDS | tr ' ' '_')
P(){ $PY -c "import sys; sys.path.insert(0, '$ROOT'); import paths; print(paths.$1)"; }
RES=$(P RES); IC=$(P IC); RUNS=${RES}runs; AN=${RES}analysis; PU2=$AN/pu2
: "${CACHE:?set CACHE}"
FT="--text-ft top --enc-lr 2e-5 --enc-dropout 0"
mkdir -p $RUNS $PU2 "$CACHE"

for step in ${*:-cache train an}; do
  case $step in
    cache)
      for ds in $DATASETS; do
        if [ $ds = kc ]; then $PY $HERE/text_electra_partial.py --dataset katchers --data-dir "${KC_TEXT_DATA:?}" --out "$CACHE/kc_kre64_L10.pt"
        else $PY $HERE/text_electra_partial.py --dataset instacart --data-dir "$IC" --out "$CACHE/ic_kre64_L10.pt"; fi
        echo "cache $ds rc=$?"
      done ;;
    train)
      for ds in $DATASETS; do
        for r in "T_textft_text text" "T_textft_text_co_lgcn_late text+graph"; do set -- $r
          D="--dump-pu $PU2/${ds}_$1_1to49_seed{seed}.json --dump-full $PU2/${ds}_$1_full_seed{seed}.json"
          if [ $ds = kc ]; then
            $PY $HERE/model_katchers.py --exp kc11/$1 --src $2 --text-path "${KC_TEXT:?}" --graph-dir "${KC_GRAPH:?}" --out-root $RUNS --seeds $SEEDS \
              --test $FT --text-cache "$CACHE/kc_kre64_L10.pt" $D > $AN/kc_$1.seeds_$TAG.log 2>&1
          else
            $PY $HERE/model_instacart.py --exp ic11/$1 --src $2 --graph-dir "$IC" --out-root $RUNS --seeds $SEEDS \
              --test $FT --enc-checkpoint --text-cache "$CACHE/ic_kre64_L10.pt" $D > $AN/ic_$1.seeds_$TAG.log 2>&1   # checkpointing: Text+Graph exceeds 24 GB without it
          fi
          echo "$ds $1 seeds $SEEDS rc=$? $(date -u +%FT%TZ)"
        done
      done ;;
    an)
      $PY $HERE/textft_an.py --datasets $DATASETS --seeds $SEEDS > $AN/textft_an.seeds_$TAG.log 2>&1; echo "textft_an rc=$?" ;;
    *) echo "unknown step $step"; exit 1 ;;
  esac
done
