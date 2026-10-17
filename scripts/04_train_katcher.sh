#!/bin/bash
# Katcher's, 5 seeds each: proposed model (Text+Graph) and the three single representations, with per-user dumps.
source "$(dirname "$0")/../config.sh"; mkdir -p $PU; G=$GR/g1/emb
kc_train I_text_co_lgcn_late text+graph $G $(dump $PU/kc_text_co_lgcn_late)
kc_train I_text              text       $G $(dump $PU/kc_text)
kc_train I_co_lgcn           graph      $G $(dump $PU/kc_co_lgcn)
kc_train K_id                id         $G $(dump $PU/kc_id_fix)
