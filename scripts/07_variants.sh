#!/bin/bash
# Variants of the proposed model (Text+Graph, real graph): item-level positions and causal mask (backbone variant, Table 12), training on the
# last order only (Appendix: training rule), mean readout of the order state (Table 12), weight 1 and 4 of the final-target term (Table A10;
# the proposed model uses 2). The last two are compiled on both datasets, as the proposed model in 04 and 05.
source "$(dirname "$0")/../config.sh"; mkdir -p $PU2
kc_train T_itempos_causal text+graph $GR/g1/emb --item-level --no-compile $(dump $PU2/kc_T_itempos_causal)
ic_train T_itempos_causal text+graph $IC        --item-level $(dump $PU2/ic_T_itempos_causal)
kc_train T_nodense        text+graph $GR/g1/emb --last-order-only --no-compile $(dump $PU2/kc_T_nodense)
ic_train T_nodense        text+graph $IC        --last-order-only $(dump $PU2/ic_T_nodense)
kc_train T_readout_mean   text+graph $GR/g1/emb --readout mean $(dump $PU2/kc_T_readout_mean)
ic_train T_readout_mean   text+graph $IC        --readout mean $(dump $PU2/ic_T_readout_mean)
for w in 1 4; do
  kc_train T_finalw$w     text+graph $GR/g1/emb --final-weight $w $(dump $PU2/kc_T_finalw$w)
  ic_train T_finalw$w     text+graph $IC        --final-weight $w $(dump $PU2/ic_T_finalw$w)
done
$PY $CODE/analysis/ablation_an.py          # paired tests against the proposed model -> $AN/ablation_an.json
