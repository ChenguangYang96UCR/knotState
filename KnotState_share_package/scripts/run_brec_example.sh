#!/usr/bin/env bash
set -euo pipefail
python expressivity/brec_knotstate_expressivity.py \
  --root ./runs/brec \
  --pair-limit 60 \
  --max-k 6 \
  --samples 2000 \
  --repeats 2 \
  --exact-limit 200000
