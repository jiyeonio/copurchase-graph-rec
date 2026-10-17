#!/bin/bash
# Baselines under full-catalogue ranking. base_full_eval.py executes the baseline script and adds the full-catalogue evaluation.
# The 1:49 and full-catalogue numbers of the baselines in the tables come from this step.
source "$(dirname "$0")/../config.sh"; B=$CODE/baselines
# Katcher's: checkpoints in $KC/checkpoints are evaluated when present, otherwise the baseline is trained with its script.
# LightGCN scores with the co-purchase graph embedding of the proposed model (f >= 1, $GR/g1/emb); the sequential baselines read only
# the number of items from the text embedding.
for m in lightgcn sasrec gru4rec bert4rec; do O=$BF/kc/$m; mkdir -p $O
  E=$(dirname $KC_TEXT); [ $m = lightgcn ] && E=$GR/g1/emb
  $PY $B/base_full_eval.py $B/original/kc_$m.py $m --data-dir $KC --emb-dir $E --ckpt-dir $KC/checkpoints --out-dir $O > $O.log 2>&1; echo "kc/$m rc=$?"
done
# Instacart: trained with the baseline script and its settings, seeds 42-46
for m in lightgcn sasrec gru4rec bert4rec; do O=$BF/ic/$m; mkdir -p $O
  $PY $B/base_full_eval.py $B/original/ic_$m.py $m --data-dir $IC --out-dir $O > $O.log 2>&1; echo "ic/$m rc=$?"
done
