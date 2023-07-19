cd ../table_model

root='./exp/'

CUDA_VISIBLE_DEVICES=0 python model_predict.py --dataset=census \
--dataset_path=$root'table/census/sampled_data.csv' \
--model_path=$root'train_dnn/train/census_base' \
--output_path=$root'train_dnn/train/census_base/predict_scores.npy' \
--output_path2=$root'train_dnn/train/census_base/labels.npy' \
2>&1 | tee ../exp/train_dnn/train/census_base/predict.log

CUDA_VISIBLE_DEVICES=0 python model_predict.py --dataset=credit \
--dataset_path=$root'table/credit/sampled_data.csv' \
--model_path=$root'train_dnn/train/credit_base' \
--output_path=$root'train_dnn/train/credit_base/predict_scores.npy' \
--output_path2=$root'train_dnn/train/credit_base/labels.npy' \
2>&1 | tee ../exp/train_dnn/train/credit_base/predict.log

CUDA_VISIBLE_DEVICES=0 python model_predict.py --dataset=bank \
--dataset_path=$root'table/bank/sampled_data.csv' \
--model_path=$root'train_dnn/train/bank_base' \
--output_path=$root'train_dnn/train/bank_base/predict_scores.npy' \
--output_path2=$root'train_dnn/train/bank_base/labels.npy' \
2>&1 | tee ../exp/train_dnn/train/bank_base/predict.log

CUDA_VISIBLE_DEVICES=0 python model_predict.py --dataset=meps \
--dataset_path=$root'table/meps/sampled_data.csv' \
--model_path=$root'train_dnn/train/meps_base' \
--output_path=$root'train_dnn/train/meps_base/predict_scores.npy' \
--output_path2=$root'train_dnn/train/meps_base/labels.npy' \
2>&1 | tee ../exp/train_dnn/train/meps_base/predict.log
