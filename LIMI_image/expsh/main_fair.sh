cd ..
baseroot='./exp/'
param1=$1
CUDA_VISIBLE_DEVICES=$param1 python main_fair.py --exp_name='celeba_gender' --exp_flag='generate' \
 --latent_file=$baseroot'img/orig_HQ100000/latent_vectors_origin.pkl' \
 --model_path=$baseroot'classifier/31_Smiling' \
 --boundary_dir=$baseroot'train_boundaries' \
 --target_attribute=31 --protected_attribute=20
