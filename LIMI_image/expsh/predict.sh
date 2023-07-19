cd ..

basepath='./exp/classifier/'

python predict_attrs.py --experiment='predict_attrs'  --exp_name='_' --attribute=31 \
    --data_dir='./exp/img/orig_HQ100000/imgs' --model_dir=$basepath'31_Smiling'
python predict_attrs.py --experiment='predict_attrs'  --exp_name='_' --attribute=20 \
    --data_dir='./exp/img/orig_HQ100000/imgs' --model_dir=$basepath'20_Male'
