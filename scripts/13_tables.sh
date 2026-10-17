#!/bin/bash
# Tables and macros: $PAPER/tables/*.tex, $PAPER/numbers.tex. Run after 12.
source "$(dirname "$0")/../config.sh"
$PY $CODE/tables/check_stats.py > $AN/check_stats.log 2>&1   # dataset statistics -> $PAPER/tables/check_stats.json
$PY $CODE/tables/make_tables.py
