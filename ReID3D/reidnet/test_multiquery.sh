# ---- Dataset paths (adjust to your actual folders) ----
TRAIN_TXT=../database/walker_2_test_200_end_full_dataset/reid3d/outputs/train_path.txt
TRAIN_INFO=../database/walker_2_test_200_end_full_dataset/reid3d/outputs/train_info.npy
TEST_TXT=../database/walker_2_test_200_end_full_dataset/reid3d/outputs/test_path.txt
TEST_INFO=../database/walker_2_test_200_end_full_dataset/reid3d/outputs/test_info.npy
QUERY_INFO=../database/walker_2_test_200_end_full_dataset/reid3d/outputs/query_IDX.npy

# ---- Where your trained checkpoint lives ----
# Example: from your cv training runs
CKPT_ROOT=walking_2_test_200_end_checkpoints_tune/lr0.0005_optAdam   # <-- adjust to chosen config
LOAD_CKPT=${CKPT_ROOT}/ckpt_best.pth

# ---- Where to save test summaries ----
EVAL_OUT=${CKPT_ROOT}/eval_multiquery
mkdir -p "${EVAL_OUT}"

python3 test_multiquery.py \
  --train_txt "${TRAIN_TXT}" \
  --train_info "${TRAIN_INFO}" \
  --test_txt "${TEST_TXT}" \
  --test_info "${TEST_INFO}" \
  --query_info "${QUERY_INFO}" \
  --n_epochs 1 \
  --optimizer AdamW \
  --lr 0.00005 \
  --lr_step_size 30 \
  --ckpt "${EVAL_OUT}" \
  --log_path loss_dummy.txt \
  --num_workers 4 \
  --class_per_batch 4 \
  --track_per_class 4 \
  --seq_len 30 \
  --test_batch 16 \
  --feat_dim 1024 \
  --stride 1 \
  --gpu_id 0 \
  --eval_freq 10 \
  --load_ckpt "${LOAD_CKPT}" \
  --n_query_rounds 20 \
  --queries_per_id 1
