cd ../
baseroot='./exp/'

step=0.3
experiment='main_fair'
### census
CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='census_gender' \
 --dataset='census' \
 --dataset_path=$baseroot'table/census/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/train/census_base' \
 --sens_param=9  --max_global=1000000  \
 --latent_file=$baseroot'table/census/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/census/census.npy' \
 --svm_file=$baseroot'train_boundaries/census/census_svm.npy' \
 --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step

CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='census_race' \
 --dataset='census' \
 --dataset_path=$baseroot'table/census/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/train/census_base' \
 --sens_param=8  --max_global=1000000  \
 --latent_file=$baseroot'table/census/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/census/census.npy' \
 --svm_file=$baseroot'train_boundaries/census/census_svm.npy' \
 --gan_file=$baseroot'gans/census/census_gan.pth' --experiment=$experiment --step=$step

CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='census_age' \
 --dataset='census' \
 --dataset_path=$baseroot'table/census/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/train/census_base' \
 --sens_param=1  --max_global=1000000  \
 --latent_file=$baseroot'table/census/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/census/census.npy' \
 --svm_file=$baseroot'train_boundaries/census/census_svm.npy' \
 --gan_file=$baseroot'gans/census/census_gan.pth'  --experiment=$experiment --step=$step

### credit locked
CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='credit_gender' \
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/train/credit_base' \
 --sens_param=9  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/credit.npy' \
 --svm_file=$baseroot'train_boundaries/credit/credit_svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step

CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='credit_age' \
 --dataset='credit' \
 --dataset_path=$baseroot'table/credit/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/train/credit_base' \
 --sens_param=13  --max_global=1000000  \
 --latent_file=$baseroot'table/credit/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/credit/credit.npy' \
 --svm_file=$baseroot'train_boundaries/credit/credit_svm.npy' \
 --gan_file=$baseroot'gans/credit/credit_gan.pth' --experiment=$experiment --step=$step

### bank

CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='bank_age' \
 --dataset='bank' \
 --dataset_path=$baseroot'table/bank/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/train/bank_base' \
 --sens_param=1  --max_global=1000000  \
 --latent_file=$baseroot'table/bank/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/bank/bank.npy' \
 --svm_file=$baseroot'train_boundaries/bank/bank_svm.npy' \
 --gan_file=$baseroot'gans/bank/bank_gan.pth' --experiment=$experiment --step=$step

### meps
CUDA_VISIBLE_DEVICES=0 python main_fair_ours.py \
 --exp_name='meps_sex' \
 --dataset='meps' \
 --dataset_path=$baseroot'table/meps/sampled_data.csv' \
 --model_path=$baseroot'train_dnn/train/meps_base' \
 --sens_param=3  --max_global=1000000  \
 --latent_file=$baseroot'table/meps/sampled_latent.pkl' \
 --boundary_file=$baseroot'train_boundaries/meps/meps.npy' \
 --svm_file=$baseroot'train_boundaries/meps/meps_svm.npy' \
 --gan_file=$baseroot'gans/meps/meps_gan.pth' --experiment=$experiment --step=$step