
baseroot='your exp dir, where stored the instances'
method_name='your method name, will be recorded in the exp log'

### attention: local_samples.npy for baselines, while global_samples.npy for LIMI

## census_gender
python eval_single.py --exp_name='census_gender' \
    --path=$baseroot'/census_gender/local_samples.npy' \
    --record='census_gender.txt' --method=$method_name \
    --dataset='census'

# census_race
python eval_single.py --exp_name='census_race' \
    --path=$baseroot'/census_race/local_samples.npy' \
    --record='census_race.txt' --method=$method_name \
    --dataset='census'

## census_age
python eval_single.py --exp_name='census_age' \
    --path=$baseroot'/census_age/local_samples.npy' \
    --record='census_age.txt' --method=$method_name \
    --dataset='census'

## credit_gender
python eval_single.py --exp_name='credit_gender' \
    --path=$baseroot'/credit_gender/local_samples.npy' \
    --record='credit_gender.txt' --method=$method_name \
    --dataset='credit'

## credit_age
python eval_single.py --exp_name='credit_age' \
    --path=$baseroot'/credit_age/local_samples.npy' \
    --record='credit_age.txt' --method=$method_name \
    --dataset='credit' 

## bank_age
python eval_single.py --exp_name='bank_age' \
    --path=$baseroot'/bank_age/local_samples.npy' \
    --record='bank_age.txt' --method=$method_name \
    --dataset='bank'


## meps_sex
python eval_single.py --exp_name='meps_sex' \
    --path=$baseroot'/meps_sex/local_samples.npy' \
    --record='meps_sex.txt' --method=$method_name \
    --dataset='meps'