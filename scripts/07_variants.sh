#!/bin/bash
# Two variants of the proposed model: item-level positions and causal mask (backbone variant), training on the last order only (Appendix: training rule).
source "$(dirname "$0")/../config.sh"; mkdir -p $PU2
kc_train T_itempos_causal text+graph $GR/g1/emb --item-level --no-compile $(dump $PU2/kc_T_itempos_causal)
ic_train T_itempos_causal text+graph $IC        --item-level $(dump $PU2/ic_T_itempos_causal)
kc_train T_nodense        text+graph $GR/g1/emb --last-order-only --no-compile $(dump $PU2/kc_T_nodense)
ic_train T_nodense        text+graph $IC        --last-order-only $(dump $PU2/ic_T_nodense)
$PY $CODE/analysis/ablation_an.py          # paired tests against the proposed model -> $AN/ablation_an.json
