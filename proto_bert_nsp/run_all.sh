#!/bin/bash
# BERT-NSP on Katcher's, seeds 42-46. GPU jobs run one after another.
#   bash proto_bert_nsp/run_all.sh                    both stages
#   bash proto_bert_nsp/run_all.sh 1                  stage 1 only: training and the 1:49 evaluation of every seed
#   bash proto_bert_nsp/run_all.sh 2                  stage 2 only: ranking of the whole catalogue with the checkpoints of stage 1
#   SEEDS="45 46" bash proto_bert_nsp/run_all.sh 2    some seeds only, e.g. on a second GPU (CUDA_VISIBLE_DEVICES=1 ...)
#   ARGS="--eval-batch 64" bash proto_bert_nsp/run_all.sh    after an out-of-memory error in evaluation (does not change the results)
# Data folders and RESULTS are those of paths.py / config.sh (KATCHERS_DATA, KATCHERS_SPLIT, RESULTS, PY). A finished seed is skipped
# and stage 2 continues an interrupted seed with the users that are missing, so the script can be started again at any time.
# After every stage the results so far are packed into <RESULTS>/proto_bert_nsp_results.tgz (without checkpoints and score matrices).
# The run is written to <RESULTS>/proto_bert_nsp/kc/bert_nsp_ndcg. If the run of the first version of this folder is next to it
# (kc/bert_nsp, checkpoint by validation loss), the seeds for which both rules chose one epoch there are taken over from it, with
# their full-catalogue ranking if it is finished, and only the other seeds are trained; that folder is read and never changed.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(dirname "$HERE")"; PY=${PY:-python}
OUT="$($PY -c "import sys; sys.path.insert(0, '$ROOT'); import paths; print(paths.RES)")proto_bert_nsp"
mkdir -p "$OUT/kc"
STAGES="${*:-1 2}"; SEEDS=${SEEDS:-42 43 44 45 46}; TAG=$(echo $SEEDS | tr ' ' '_'); NAME=bert_nsp_ndcg

$PY "$HERE/check_protocol.py" > "$OUT/kc/check_protocol.log" 2>&1; rc=$?; echo "check_protocol rc=$rc"
[ $rc -ne 0 ] && { echo "the data are not read as the baselines read them, or the two inputs differ: see $OUT/kc/check_protocol.log"; exit 1; }

pack(){ # results without checkpoints and score matrices
  $PY "$HERE/run.py" collect --name $NAME > /dev/null 2>&1
  $PY "$HERE/make_numbers.py" --name $NAME > "$OUT/make_numbers.log" 2>&1; echo "make_numbers rc=$?"
  tar czf "$OUT/../proto_bert_nsp_results.tgz" --exclude checkpoints --exclude '*.npy' -C "$OUT/.." proto_bert_nsp \
    && echo "packed: $(cd "$OUT/.." && pwd)/proto_bert_nsp_results.tgz"; }

for stage in $STAGES; do
  case $stage in
    1) $PY "$HERE/run.py" train --name $NAME --seeds $SEEDS $ARGS >> "$OUT/kc/$NAME.train.seeds_$TAG.log" 2>&1; rc=$?; echo "stage 1 (training, 1:49) rc=$rc" ;;
    2) $PY "$HERE/run.py" full  --name $NAME --seeds $SEEDS $ARGS >> "$OUT/kc/$NAME.full.seeds_$TAG.log" 2>&1; rc=$?; echo "stage 2 (full catalogue) rc=$rc" ;;
    *) echo "unknown stage $stage"; exit 1 ;;
  esac
  pack
  if [ $rc -ne 0 ]; then echo "stage $stage stopped: see the end of its log in $OUT/kc/"; exit 1; fi
done
exit 0
