#!/bin/bash
# Appendix comparison experiment (an English text encoder for Instacart): Text only and Text+Graph with the ELECTRA-base embedding. Run after 03.
source "$(dirname "$0")/../config.sh"; mkdir -p $PU_ENC
for x in "text_electra text" "text_co_lgcn_late_electra text+graph"; do set -- $x
  $PY $CODE/model/model_instacart.py --exp ic11/Z_$1 --src $2 --text-path $IC_TEXT_EN --graph-dir $IC --out-root $RUNS_ENC --seeds $SEEDS --test \
    $(dump $PU_ENC/ic_$1) > $PU_ENC/../ic_Z_$1.log 2>&1; echo "Z_$1 rc=$?"
done
