cd ..

for i in 20 31
do
echo $i
python train_attr_classifier.py --exp_name='_' --attribute=$i

done