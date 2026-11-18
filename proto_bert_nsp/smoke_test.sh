#!/bin/bash
# Checks that the code runs on this machine, on small random data in a temporary folder. Touches neither the data of the paper nor
# results/. The numbers it prints mean nothing.
#   part 1  a small BERT with random weights: both stages, an interrupted stage 2, the macros (CPU is enough)
#   part 2  the pretrained model (downloaded on first use, about 0.7 GB): one epoch and the ranking of three users on the default device
set -e -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PY=${PY:-python}; T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
$PY "$HERE/make_fake_data.py" --out "$T/data"
D="--data-dir $T/data/kc --split-dir $T/data/split"
echo "check_protocol: $($PY "$HERE/check_protocol.py" $D --emb-dir "$T/data/kc_emb" | tail -1)"

run(){ # log-name, then the arguments of run.py
  local log="$T/$1.log"; shift
  $PY "$HERE/run.py" "$@" $D > "$log" 2>&1 || { tail -30 "$log"; echo "FAILED: run.py $*"; exit 1; }; }

# ---- part 1
A="--tiny --out-dir $T/out/tiny --epochs 2 --device cpu"
run t1 train $A --seeds 42 43;            echo "tiny: $(grep ' seed 43\] epoch' "$T/t1.log")"
run t2 full  $A --seeds 42 43;            echo "tiny: $(grep ' seed 43\] full masked' "$T/t2.log")"
before="$($PY -c "import json; print(json.load(open('$T/out/tiny/full_seed43.json'))['row']['full_masked_ndcg@10_overall'])")"
$PY - "$T/out/tiny" <<'EOF'
import os, sys, numpy as np
out = sys.argv[1]                                   # as if stage 2 of seed 43 had been interrupted after 25 users
s = np.load(os.path.join(out, "scores_full_seed43.npy"), mmap_mode="r+"); s[25:] = np.nan; s.flush()
os.remove(os.path.join(out, "full_seed43.json"))
EOF
run t3 full  $A --seeds 43
grep -q 'full catalogue: 25 of' "$T/t3.log" || { echo "FAILED: the interrupted stage 2 did not continue"; exit 1; }
after="$($PY -c "import json; print(json.load(open('$T/out/tiny/full_seed43.json'))['row']['full_masked_ndcg@10_overall'])")"
[ "$before" = "$after" ] || { echo "FAILED: the continued stage 2 gives $after, the uninterrupted one $before"; exit 1; }
echo "tiny: interrupted stage 2 continued, same result"
$PY "$HERE/make_numbers.py" --results "$T/out" --name tiny --paper-pu "$T/none/" > "$T/numbers.log" 2>&1 || { tail -20 "$T/numbers.log"; echo "FAILED: make_numbers.py"; exit 1; }
echo "tiny: make_numbers $(tail -1 "$T/numbers.log" | cut -d: -f1)"

# ---- part 2
B="--out-dir $T/out/real --epochs 1 --seeds 42"
run r1 train $B;                          echo "pretrained: $(grep ' seed 42\] epoch' "$T/r1.log")"
run r2 full  $B --limit-users 3;          echo "pretrained: $(grep ' seed 42\] full masked' "$T/r2.log")"
echo "device: $(grep '"device"' "$T/r1.log" | head -1 | tr -d ' ",')"
echo "SMOKE TEST OK"
