#!/usr/bin/env bash
set -euo pipefail
# Replace the example paths with your precomputed official WLKS Gram matrices
# and KnotState feature cache.
python fusion/wlks_knotstate_fusion.py \
  --wlks wl2=/path/to/WLKS_layer2.npy \
  --knot /path/to/knot_r1_m2048_seed2_exact2000.npy \
  --labels /path/to/labels.npy \
  --splits /path/to/splits.npy \
  --orders 3,4,6 \
  --out ./runs/fusion
