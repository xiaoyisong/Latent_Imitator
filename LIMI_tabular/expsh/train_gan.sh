cd ../
CUDA_VISIBLE_DEVICES=0 python train_gan.py --exp_name=census \
    --num_samples=100 --save='cencus_gan.pth' \
    --output='cencus_sample.csv' \
    --output_train='cencus_train_raw.csv' \
    --data='./datasets/census_train.csv'

CUDA_VISIBLE_DEVICES=0 python train_gan.py --exp_name=credit \
    --num_samples=100 --save='credit_gan.pth' \
    --output='credit_sample.csv' \
    --output_train='credit_train_raw.csv' \
    --data='./datasets/credit_train.csv'

CUDA_VISIBLE_DEVICES=0 python train_gan.py --exp_name=bank \
    --num_samples=100 --save='bank_gan.pth' \
    --output='bank_sample.csv' \
    --output_train='bank_train_raw1w.csv' \
    --data='./datasets/bank_train.csv'

CUDA_VISIBLE_DEVICES=0 python train_gan.py --exp_name=meps \
    --num_samples=100 --save='meps_gan.pth' \
    --output='meps_sample.csv' \
    --output_train='meps_train_raw.csv' \
    --data='./datasets/meps_train.csv'    

## the gans are stored in ../exp/gans manually