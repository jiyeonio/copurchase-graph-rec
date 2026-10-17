#!/bin/bash
# Analyses behind the tables and figures. Run after 04-11.
source "$(dirname "$0")/../config.sh"; mkdir -p $RK $S/extra_kre $S/extra; A=$CODE/analysis
# (1) ranks under full-catalogue ranking without removing history items (Appendix), from the existing checkpoints
for x in "id id K_id" "text text I_text" "co_lgcn graph I_co_lgcn" "text_co_lgcn_late text+graph I_text_co_lgcn_late"; do set -- $x
  kc_train R_$1 $2 $GR/g1/emb --eval-only --ckpt-dir $RUNS/kc11/$3 --no-compile --dump-rank $RK/kc_$1_{mask}_seed{seed}.json
  ic_train R_$1 $2 $IC        --eval-only --ckpt-dir $RUNS/ic11/$3 --no-compile --dump-rank $RK/ic_$1_{mask}_seed{seed}.json
done
# (2) test users by target group and by degree, real graph against the rewired control
$PY $A/link_an_lg.py
# (3) single statements: order size, degree and popularity, link position and strength, edge removal, text encoders
$PY $A/extra_an2.py; $PY $A/extra_ordsize.py; $PY $A/story_an.py; $PY $A/edge_removal_an.py; $PY $A/tok_stats.py
# (4) nearest neighbours of the two fixed representations and the t-SNE figure (needs matplotlib: PYLIB = a folder that contains it)
$PY $A/kc_item_meta.py
PYTHONPATH=$PYLIB $PY $A/emb_vis2.py
# (5) case-study figure (seed 42 of the proposed model on Instacart)
ic_train R_case text+graph $IC --seeds 42 --eval-only --ckpt-dir $RUNS/ic11/I_text_co_lgcn_late --no-compile --dump-case $AN/case_ic_seed{seed}.json
$PY $A/make_case.py
