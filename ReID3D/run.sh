#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for SEED in 0 1 2 3 4; do
  
  DATA_DIR="$ROOT/database/seed_${SEED}_walking_2_test_200_end/reid3d"
  OUT_DIR="$DATA_DIR/outputs"
  INFO_DIR="$DATA_DIR/info"
  
  echo "SEED ${SEED} DATABASE"
  
  mkdir -p "$OUT_DIR"
  
  python "$ROOT/database/create_reid_database.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR" \
    --info_dir "$INFO_DIR"
done