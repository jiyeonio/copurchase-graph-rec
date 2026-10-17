#!/bin/bash
# Rewiring control (main comparison and the appendix table for all groups): the graph representation is replaced by the LightGCN embedding of the
# rewired graph and the model is retrained, for the graph alone and for Text+Graph.
source "$(dirname "$0")/../config.sh"; mkdir -p $PU2
kc_train X_co_lgcn_rwlg           graph      $GR/g1_rwlg/emb --no-compile $(dump $PU2/kc_co_lgcn_rwlg)
kc_train X_text_co_lgcn_late_rwlg text+graph $GR/g1_rwlg/emb --no-compile $(dump $PU2/kc_text_co_lgcn_late_rwlg)
ic_train X_co_lgcn_rwlg           graph      $IG/rw_co_lg/emb $(dump $PU2/ic_co_lgcn_rwlg)
ic_train X_text_co_lgcn_late_rwlg text+graph $IG/rw_co_lg/emb $(dump $PU2/ic_text_co_lgcn_late_rwlg)
