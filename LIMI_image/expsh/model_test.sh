cd ..
baseroot='./exp/'
param1=$1

CUDA_VISIBLE_DEVICES=$param1 python model_test.py --exp_name='smile_gender_cz0' \
 --data_dir=$baseroot'main_fair/celeba_gender/imgs_p1/cz0' \
 --data_pair_dir=$baseroot'main_fair/celeba_gender/imgs_p2/cz0' \
 --attribute=31 --protected_attribute=20 \
 --model_dir=$baseroot'classifier_celeba/31_Smiling' \
 --model_dir2=$baseroot'classifier_celeba/20_Male' \
 --test_iname='cz0'

CUDA_VISIBLE_DEVICES=$param1 python model_test.py --exp_name='smile_gender_cp1' \
 --data_dir=$baseroot'main_fair/celeba_gender/imgs_p1/cp1' \
 --data_pair_dir=$baseroot'main_fair/celeba_gender/imgs_p2/cp1' \
 --attribute=31 --protected_attribute=20 \
 --model_dir=$baseroot'classifier_celeba/31_Smiling' \
 --model_dir2=$baseroot'classifier_celeba/20_Male' \
 --test_iname='cp1'

CUDA_VISIBLE_DEVICES=$param1 python model_test.py --exp_name='smile_gender_cn1' \
 --data_dir=$baseroot'main_fair/celeba_gender/imgs_p1/cn1' \
 --data_pair_dir=$baseroot'main_fair/celeba_gender/imgs_p2/cn1' \
 --attribute=31 --protected_attribute=20 \
 --model_dir=$baseroot'classifier_celeba/31_Smiling' \
 --model_dir2=$baseroot'classifier_celeba/20_Male' \
 --test_iname='cn1'
