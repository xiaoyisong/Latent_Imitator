cd ../
cuda_id=0
CUDA_VISIBLE_DEVICES=$cuda_id python train_dnn.py --exp_name='census_base' \
    --dataset='census' \
    --exp_flag='train'

CUDA_VISIBLE_DEVICES=$cuda_id python train_dnn.py --exp_name='credit_base' \
    --dataset='credit' \
    --exp_flag='train'

CUDA_VISIBLE_DEVICES=$cuda_id python train_dnn.py --exp_name='bank_base' \
    --dataset='bank' \
    --exp_flag='train'

CUDA_VISIBLE_DEVICES=$cuda_id python train_dnn.py --exp_name='meps_base' \
    --dataset='meps' \
    --exp_flag='train'



