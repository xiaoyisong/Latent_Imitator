cd ..

predictpath='./exp/predict_attrs/'

latentroot='./exp/img/orig_HQ100000/'
python train_latent_boundary.py --experiment='train_boundaries'  --exp_name='_' --attribute=31 \
    --latent_file=$latentroot'latent_vectors_origin.pkl' \
    --score_file=$predictpath'31_Smiling_rawScore.npy' --train_num=5000 \
    --label_file=$predictpath'31_Smiling.npy'
python train_latent_boundary.py --experiment='train_boundaries'  --exp_name='_' --attribute=20 \
    --latent_file=$latentroot'latent_vectors_origin.pkl' \
    --score_file=$predictpath'20_Male_rawScore.npy' --train_num=5000 \
    --label_file=$predictpath'20_Male.npy'