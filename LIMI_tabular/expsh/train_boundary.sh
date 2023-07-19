cd ..

root='./exp/'

# census
python train_latent_boundary.py \
 --exp_name='census' \
 --latent_file=$root'table/census/sampled_latent.pkl' \
 --label_file=$root'train_dnn/train/census_base/labels.npy' \
 --score_file=$root'train_dnn/train/census_base/predict_scores.npy' \
 --train_num=50000

# credit
python train_latent_boundary.py \
 --exp_name='credit' \
 --latent_file=$root'table/credit/sampled_latent.pkl' \
 --label_file=$root'train_dnn/train/credit_base/labels.npy' \
 --score_file=$root'train_dnn/train/credit_base/predict_scores.npy' \
 --train_num=50000

## bank
python train_latent_boundary.py \
 --exp_name='bank' \
 --latent_file=$root'table/bank/sampled_latent.pkl' \
 --label_file=$root'train_dnn/train/bank_base/labels.npy' \
 --score_file=$root'train_dnn/train/bank_base/predict_scores.npy' \
 --train_num=50000


## meps
python train_latent_boundary.py \
 --exp_name='meps' \
 --latent_file=$root'table/meps/sampled_latent.pkl' \
 --label_file=$root'train_dnn/train/meps_base/labels.npy' \
 --score_file=$root'train_dnn/train/meps_base/predict_scores.npy' \
 --train_num=50000
