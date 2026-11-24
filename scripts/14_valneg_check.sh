#!/bin/bash
# Katcher's validation negatives (Section 5.1): (1) how many of the 49 validation negatives are items of the user's history; (2) the four
# models of 04 trained again with the validation negatives drawn outside the history (--val-exclude-history; all else as in 04: compiled,
# defaults of the paper, seeds 42-46), compared with the runs of 04 per user, also in the three target groups of Table 8. Run after 04 and 12.
source "$(dirname "$0")/../config.sh"; mkdir -p $PU2; G=$GR/g1/emb
$PY $CODE/analysis/val_neg_history.py > $AN/val_neg_history.log 2>&1          # -> $AN/val_neg_history.json
kc_train I_text_co_lgcn_late_valnh text+graph $G --val-exclude-history $(dump $PU2/kc_text_co_lgcn_late_valnh)
kc_train I_text_valnh              text       $G --val-exclude-history $(dump $PU2/kc_text_valnh)
kc_train I_co_lgcn_valnh           graph      $G --val-exclude-history $(dump $PU2/kc_co_lgcn_valnh)
kc_train K_id_valnh                id         $G --val-exclude-history $(dump $PU2/kc_id_fix_valnh)
$PY $CODE/analysis/valneg_check_an.py > $AN/valneg_check_an.log 2>&1           # -> $AN/valneg_check_an.json
