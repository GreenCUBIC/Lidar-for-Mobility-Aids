for MODE in walking wheelchair rollatorshuffling rollatorwalking walker cane crutches; do 
  for SEED in 0 1 2 3 4; do
  
    # ---- Dataset paths ----
    TRAIN_TXT=../database/seed_${SEED}_${MODE}_2_test_200_end/reid3d/outputs/train_path.txt
    TRAIN_INFO=../database/seed_${SEED}_${MODE}_2_test_200_end/reid3d/outputs/train_info.npy
    TEST_TXT=../database/seed_${SEED}_${MODE}_2_test_200_end/reid3d/outputs/test_path.txt
    TEST_INFO=../database/seed_${SEED}_${MODE}_2_test_200_end/reid3d/outputs/test_info.npy
    QUERY_INFO=../database/seed_${SEED}_${MODE}_2_test_200_end/reid3d/outputs/query_IDX.npy
    
    # ---- Where your trained checkpoint lives ----
    # Example: from your cv training runs
    CKPT_ROOT=${MODE}_2_person_200_end_PRETRAINEDckpt_cv_5e4/class_${MODE}_200_end/seed_${SEED}/lr0.0005_optAdamW   # <-- adjust to chosen config
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
      --lr 0.0005 \
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
  
  done
done