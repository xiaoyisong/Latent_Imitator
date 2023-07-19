import logging
import torch
import torch.nn as nn
import torchvision
import numpy as np
from Models.attr_classifier import attribute_classifier
from utils import parse_args
import pickle
from utils import util
import argparse
from data.celeba import create_dataset_synthesis
import os
from utils.logger import setup_logger

EXP_DIR = './exp'


def _collect_args_predict_attrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='predict_attrs')
    parser.add_argument('--exp_name', type=str, default='_')
    parser.add_argument('--data_dir', type=str, default='_')
    parser.add_argument('--attribute', type=int, default=31)
    parser.add_argument('--model_dir', type=str, default='_')
    parser.add_argument('--random_seed', type=int, default=2333)
    parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())
    attr_list = util.get_all_attr()
    attr_name = attr_list[opt['attribute']]
    opt['attr_name'] = attr_name
    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
    opt['dtype'] = torch.float32

    params_test = {'batch_size': 128, 'shuffle': False, 'num_workers': 4}
    data_setting = {
        'data_dir': opt['data_dir'],
        'params_test': params_test,
    }
    opt['data_setting'] = data_setting

    if opt['exp_name'] == '_':
        opt['exp_name'] = str(opt['attribute']) + '_' + attr_name

    opt['expdir'] = os.path.join(EXP_DIR, opt['experiment'])
    util.make_dir(opt['expdir'])

    opt['model_path'] = os.path.join(opt['model_dir'], 'best.pth')
    opt['thr_path'] = os.path.join(opt['model_dir'], 'val_results.pkl')
    opt['output_file'] = os.path.join(opt['expdir'], opt['exp_name'] + '.npy')
    opt['output_file2'] = os.path.join(opt['expdir'], opt['exp_name'] + '_rawScore.npy')

    logger = setup_logger(opt['expdir'], logger_name='logger', mode='a+')

    # Uncomment if deterministic run required
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opt['random_seed'])
    np.random.seed(opt['random_seed'])
    logger.info(f"the random_seed is set as {opt['random_seed']}\n")
    logger.info(f"setting {opt}\n")

    return opt


if __name__ == '__main__':

    opt = _collect_args_predict_attrs()
    logger = logging.getLogger('logger')

    data_setting = opt['data_setting']
    logger.info(f"data_setting is {data_setting}")

    test_loader = create_dataset_synthesis(
        data_dir=data_setting['data_dir'],
        params=data_setting['params_test'],
        augment=False,
    )

    AC = attribute_classifier(opt['device'], opt['dtype'], modelpath=opt['model_path'])

    logger.info(
        f"predict attr {opt['attr_name']} using {opt['model_path'].split('/')[-2]} model"
    )

    _, scores = AC.get_scores(test_loader, False)

    threshold = pickle.load(open(opt['thr_path'], 'rb'))['f1_thresh']
    scores_bi = np.where(scores > threshold, 1.0, 0.0)

    with open(opt['output_file'], 'wb+') as handle:
        pickle.dump(scores_bi, handle)
    with open(opt['output_file2'], 'wb+') as handle:
        pickle.dump(scores, handle)
