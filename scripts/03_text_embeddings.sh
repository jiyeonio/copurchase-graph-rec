#!/bin/bash
# Text representation of the items: fixed 768-d [CLS] embedding of the item description, no fine-tuning.
# Both datasets: snunlp/KR-ELECTRA-discriminator, [CLS], max_length 64 (graphs/text_electra.py). An existing file is not overwritten.
source "$(dirname "$0")/../config.sh"
[ -f "$KC_TEXT" ] || $PY $CODE/graphs/text_electra.py --dataset katchers --data-dir $KC_META --emb-dir "$(dirname "$KC_TEXT")"
[ -f "$IC_TEXT" ] || $PY $CODE/graphs/text_electra.py --dataset instacart --data-dir $IC --emb-dir "$(dirname "$IC_TEXT")"
# Appendix comparison experiment: Instacart with google/electra-base-discriminator
[ -f "$IC_TEXT_EN" ] || $PY $CODE/graphs/text_electra_en.py
