#!/bin/bash
# Appendix (co-purchase threshold on Katcher's): graphs with f >= 2 and f >= 3 (f >= 1 is the main setting).
source "$(dirname "$0")/../config.sh"
for f in 2 3; do g=$GR/gf$f/emb
  kc_train S_co_f$f graph $g --no-compile
  kc_train S_text_co_f${f}_late text+graph $g --no-compile
done
