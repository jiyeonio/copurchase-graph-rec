#!/bin/bash
# Instacart, 5 seeds each (about 10 minutes per seed): proposed model and the three single representations, with per-user dumps.
source "$(dirname "$0")/../config.sh"; mkdir -p $PU; G=$IC
ic_train I_text_co_lgcn_late text+graph $G $(dump $PU/ic_text_co_lgcn_late)
ic_train I_text              text       $G $(dump $PU/ic_text)
ic_train I_co_lgcn           graph      $G $(dump $PU/ic_co_lgcn)
ic_train K_id                id         $G $(dump $PU/ic_id_fix)
