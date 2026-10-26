#!/bin/bash

# Run command adapted from your previous settings
# Key changes:
# - Uses improved_processing.py with better corner detection
# - Slightly adjusted parameters based on your data characteristics
# - Keeps your train/test split configuration

#rm -r data/seankirkby/ReID3D/database/hyp_tune

#!/bin/bash
set -e

CSV_ROOT=/data/seankirkby/data
BASE_OUT=/data/seankirkby/ReID3D/database

CLASS=0
TRAIN_PERSONS=28
TEST_PERSONS=2
FRAME_START=200
#FRAME_END= 600

for SEED in 0 1 2 3 4; do
  OUT_ROOT=${BASE_OUT}/seed_${SEED}_walking_${TEST_PERSONS}_test_${FRAME_START}_end
  #${FRAME_END}

  echo "=========================================="
  echo "[SEED ${SEED}] out-root = ${OUT_ROOT}"
  echo "=========================================="

  python improved_processing.py \
    --csv-root ${CSV_ROOT} \
    --out-root ${OUT_ROOT} \
    --csv-has-header \
    --x-col 'Points_m_XYZ:0' \
    --y-col 'Points_m_XYZ:1' \
    --z-col 'Points_m_XYZ:2' \
    --include-classes ${CLASS} \
    --force-split train_test \
    --train-persons ${TRAIN_PERSONS} \
    --test-persons ${TEST_PERSONS} \
    --split-seed ${SEED} \
    --dbscan-eps 0.22 \
    --dbscan-min-samples 40 \
    --min-person-points 100 \
    --person-h 0.8 2.0 \
    --person-w 0.30 1.5 \
    --person-d 0.30 1.5 \
    --use-scoring \
    --max-person-per-frame 1 \
    --near-wall-band 0.40 \
    --near-wall-frac-max 0.30 \
    --planarity-max 0.90 \
    --sphericity-min 0.02 \
    --corner-threshold 0.9 \
    --preclear-margin 0.0 \
    --intra-clearance 0.02 \
    --intra-outlier-nb 16 \
    --intra-outlier-std 2.0 \
    --max-walls 2 \
    --wall-dist 0.02 \
    --dims-strong-gate \
    --dims-gate-slack 0.20 \
    --max-xy-radius 0 \
    --max-person-points 1000 \
    --max-person-penalty 100.0 \
    --max-person-penalty-shape linear \
    --frame-start ${FRAME_START} 
    #--frame-end ${FRAME_END}
done

  #--verbose

  # --export-topk 0\
  # --keep-top1 \
  
# Key changes for aggressive corner rejection:
# 1. --corner-threshold 0.25 (lower = more aggressive rejection)
# 2. --near-wall-frac-max 0.30 (stricter - corners are always near walls)
# 3. --planarity-max 0.70 (lower - corners are planar)
# 4. --sphericity-min 0.05 (higher - corners lack volume)
# 5. --near-wall-band 0.05 (tighter band for detection)
# 6. Removed --keep-top1 flag (don't force keeping largest if it's a corner)

#python inspect_bin_folder.py bigger_out_overfit_class/reid3d/ --recursive
if [ -f notification.py ]; then python notification.py; fi