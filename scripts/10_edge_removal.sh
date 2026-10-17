#!/bin/bash
# Appendix (removal of edges): Text+Graph on Katcher's with 75 / 50 / 25% of the co-purchase edges kept. Run after 01.
source "$(dirname "$0")/../config.sh"; mkdir -p $PU_EDGE
for d in 25 50 75; do n=text_co_drop${d}_late
  $PY $CODE/model/model_katchers.py --exp kc11/B2_$n --src text+graph --text-path $KC_TEXT --graph-dir $GR/gdrop$d/emb --out-root $RUNS_EDGE --seeds $SEEDS --test \
    $(dump $PU_EDGE/kc_$n) > $PU_EDGE/../kc_B2_$n.log 2>&1; echo "B2_$n rc=$?"
done
$PY $CODE/analysis/edge_removal_an.py      # -> $S/extra_kre/boost23.json
