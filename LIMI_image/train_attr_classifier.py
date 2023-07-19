import logging
import os
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torch.nn.functional as F
import numpy as np
from os import listdir, path, mkdir
from PIL import Image
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from Models.attr_classifier import attribute_classifier
import argparse
from utils import util
from utils.logger import setup_logger
import pickle

from data.celeba import create_dataset_celeba

EXP_DIR = './exp'


def _collect_args_classifier():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='classifier')
    parser.add_argument('--exp_name', type=str, default='_')
    parser.add_argument(
        '--data_dir', type=str, default='./celebA/img_align_celeba_crop',
    )
    parser.add_argument(
        '--split_file', type=str, default='./celebA/Eval/list_eval_partition.txt',
    )
    parser.add_argument(
        '--attr_file', type=str, default='./celebA/Anno/list_attr_celeba.txt',
    )
    parser.add_argument('--random_seed', type=int, default=2333)
    parser.add_argument('--attribute', type=int, default=31)
    parser.add_argument('--protected_attribute', type=int, default=20)
    parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())

    attr_list = util.get_all_attr()
    attr_name = attr_list[opt['attribute']]
    print(attr_name)
    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
    opt['dtype'] = torch.float32
    opt['print_freq'] = 200
    opt['total_epochs'] = 20

    optimizer_setting = {
        'optimizer': torch.optim.Adam,
        'lr': 1e-4,
        'weight_decay': 0,
    }
    opt['optimizer_setting'] = optimizer_setting

    params_train = {'batch_size': 128, 'shuffle': True, 'num_workers': 4}
    params_val = {'batch_size': 128, 'shuffle': False, 'num_workers': 4}
    data_setting = {
        'data_dir': opt['data_dir'],
        'split_file': opt['split_file'],
        'attr_file': opt['attr_file'],
        'params_train': params_train,
        'params_val': params_val,
        'protected_attribute': opt['protected_attribute'],
        'attribute': opt['attribute'],
        'augment': True,
    }
    opt['data_setting'] = data_setting

    if opt['exp_name'] == '_':
        opt['exp_name'] = attr_name

    opt['expdir'] = os.path.join(
        EXP_DIR + '/classifier', str(opt['attribute']) + '_' + opt['exp_name']
    )
    util.make_dir(opt['expdir'])

    logger = setup_logger(opt['expdir'], logger_name='logger')

    # Uncomment if deterministic run required
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opt['random_seed'])
    np.random.seed(opt['random_seed'])
    logger.info(f"the random_seed is set as {opt['random_seed']}\n")
    logger.info(f"setting {opt}\n")
    return opt


def test_model(model, loader, opt, mode='train'):
    targets, scores = model.get_scores(loader)

    os.path.join(opt['expdir'], mode + '_scores.pkl')
    with open(os.path.join(opt['expdir'], mode + '_scores.pkl'), 'wb+') as handle:
        pickle.dump(scores, handle)
    with open(os.path.join(opt['expdir'], mode + '_targets.pkl'), 'wb+') as handle:
        pickle.dump(targets, handle)

    cal_thresh = util.calibrated_threshold(targets[:, 0], scores)
    f1_score, f1_thresh = util.get_threshold(targets[:, 0], scores)

    val_results = {
        'f1_thresh': f1_thresh,
        'cal_thresh': cal_thresh,
    }

    with open(os.path.join(opt['expdir'], mode + '_results.pkl'), 'wb+') as handle:
        pickle.dump(val_results, handle)


def train_attr_classifier(opt):
    logger.info(f"train classifier")
    logger.info(f"{opt['data_setting']}")
    data_setting = opt['data_setting']

    train_loader = create_dataset_celeba(
        data_dir=data_setting['data_dir'],
        attr_file=data_setting['attr_file'],
        attribute=data_setting['attribute'],
        protected_attribute=data_setting['protected_attribute'],
        params=data_setting['params_train'],
        augment=data_setting['augment'],
        split='train',
    )

    val_loader = create_dataset_celeba(
        data_dir=data_setting['data_dir'],
        attr_file=data_setting['attr_file'],
        attribute=data_setting['attribute'],
        protected_attribute=data_setting['protected_attribute'],
        params=data_setting['params_val'],
        augment=False,
        split='val',
    )

    test_loader = create_dataset_celeba(
        data_dir=data_setting['data_dir'],
        attr_file=data_setting['attr_file'],
        attribute=data_setting['attribute'],
        protected_attribute=data_setting['protected_attribute'],
        params=data_setting['params_val'],
        augment=False,
        split='test',
    )

    save_path = os.path.join(opt['expdir'], 'best.pth')
    save_path_curr = os.path.join(opt['expdir'], 'current.pth')
    model_path = None
    if path.exists(save_path_curr):
        logger.info(f'Model exists, resuming training')
        model_path = save_path_curr
    AC = attribute_classifier(opt['device'], opt['dtype'], modelpath=model_path)
    for i in range(AC.epoch, opt['total_epochs']):
        AC.train(train_loader)
        acc = AC.check_avg_precision(val_loader, weights=None)

        if acc > AC.best_acc:
            AC.best_acc = acc
            AC.save_model(save_path)
        AC.save_model(save_path_curr)

    AC = attribute_classifier(opt['device'], opt['dtype'], modelpath=save_path)
    test_model(model=AC, loader=val_loader, opt=opt, mode='val')
    test_model(model=AC, loader=test_loader, opt=opt, mode='test')


if __name__ == '__main__':
    opt = _collect_args_classifier()
    logger = logging.getLogger('logger')
    train_attr_classifier(opt=opt)
