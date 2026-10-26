TRAIN_TXT=../database/walker_2_test_400_650_full_dataset/reid3d/outputs/train_path.txt
TRAIN_INFO=../database/walker_2_test_400_650_full_dataset/reid3d/outputs/train_info.npy
TEST_TXT=../database/walker_2_test_400_650_full_dataset/reid3d/outputs/test_path.txt
TEST_INFO=../database/walker_2_test_400_650_full_dataset/reid3d/outputs/test_info.npy
QUERY_INFO=../database/walker_2_test_400_650_full_dataset/reid3d/outputs/query_IDX.npy
BASE_CKPT=checkpoints/PRETRAINED_ARCHIVE_ckpt_best.pth


python3 full_train.py \
    --train_txt $TRAIN_TXT --train_info $TRAIN_INFO  --test_batch 16 \
    --test_txt $TEST_TXT  --test_info $TEST_INFO --query_info $QUERY_INFO \
    --n_epochs 750 --lr 0.00005 --lr_step_size 30 --optimizer AdamW \
    --ckpt log_dummy --log_path loss.txt --num_workers 4 \
    --class_per_batch 4 --track_per_class 4 --seq_len 30 \
    --feat_dim 1024 --stride 1 --gpu_id '0' --eval_freq 10 #\
    #--load_ckpt $BASE_CKPT
