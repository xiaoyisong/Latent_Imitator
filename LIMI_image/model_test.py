import logging
import os
import time
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
from utils.logger import setup_logger

EXP_DIR = './exp'


def collect_args_fair_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='model_test')
    parser.add_argument('--exp_name', type=str, default='_')
    parser.add_argument('--data_dir', type=str, default='_')
    parser.add_argument('--data_pair_dir', type=str, default='_')
    parser.add_argument('--attribute', type=int, default=31)
    parser.add_argument('--protected_attribute', type=int, default=20)
    parser.add_argument('--number', type=int, default=None)
    parser.add_argument('--model_dir', type=str, default='_')
    parser.add_argument('--model_dir2', type=str, default='_')
    parser.add_argument('--test_iname', type=str, default='cz0')
    parser.add_argument('--random_seed', type=int, default=2333)
    parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())
    attr_list = util.get_all_attr()
    attr_name = attr_list[opt['attribute']]
    opt['attr_name'] = attr_name

    attr_name = attr_list[opt['protected_attribute']]
    opt['protected_attribute_name'] = attr_name

    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
    opt['dtype'] = torch.float32

    params_test = {'batch_size': 128, 'shuffle': False, 'num_workers': 4}
    data_setting = {
        'data_dir': opt['data_dir'],
        'data_pair_dir': opt['data_pair_dir'],
        'number': opt['number'],
        'params_test': params_test,
    }
    opt['data_setting'] = data_setting

    if opt['exp_name'] == '_':
        opt['exp_name'] = str(opt['attribute']) + '_' + attr_name

    opt['expdir'] = os.path.join(EXP_DIR, opt['experiment'], opt['exp_name'])
    util.make_dir(opt['expdir'])

    opt['model_path'] = os.path.join(opt['model_dir'], 'best.pth')
    opt['thr_path'] = os.path.join(opt['model_dir'], 'val_results.pkl')

    opt['model_path2'] = os.path.join(opt['model_dir2'], 'best.pth')
    opt['thr_path2'] = os.path.join(opt['model_dir2'], 'val_results.pkl')

    logger = setup_logger(
        opt['expdir'],
        logfile_name='test_' + opt['test_iname'] + '.txt',
        logger_name='logger',
        mode='w',
    )

    # Uncomment if deterministic run required
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opt['random_seed'])
    np.random.seed(opt['random_seed'])
    logger.info(f"the random_seed is set as {opt['random_seed']}\n")
    logger.info(f"setting {opt}\n")

    return opt, logger


class ModelTest:
    def __init__(self, opt, logger) -> None:
        self.opt = opt
        self.logger = logger
        self.build_data()
        self.build_test_model()

    def build_data(self):
        data_setting = self.opt['data_setting']
        self.logger.info(f"data_setting is {data_setting}")
        self.test_loader = create_dataset_synthesis(
            data_dir=data_setting['data_dir'],
            number=data_setting['number'],
            params=data_setting['params_test'],
            augment=False,
            iname=self.opt['test_iname'],
        )

        self.pair_loader = create_dataset_synthesis(
            data_dir=data_setting['data_pair_dir'],
            number=data_setting['number'],
            params=data_setting['params_test'],
            augment=False,
            iname=self.opt['test_iname'],
        )
        self.data_setting = data_setting

    def build_test_model(self):

        self.target_model = attribute_classifier(
            self.opt['device'], self.opt['dtype'], modelpath=self.opt['model_path']
        )
        self.target_threshold = pickle.load(open(self.opt['thr_path'], 'rb'))[
            'f1_thresh'
        ]
        self.logger.info(f"target_threshold value : {self.target_threshold}")

        self.protect_model = attribute_classifier(
            self.opt['device'], self.opt['dtype'], modelpath=self.opt['model_path2']
        )
        self.protect_threshold = pickle.load(open(self.opt['thr_path2'], 'rb'))[
            'f1_thresh'
        ]
        self.logger.info(f"protect_threshold value : {self.protect_threshold}")

    def run_test(self,):

        ## target
        self.logger.info(
            f"predict attr {self.opt['attr_name']} using {self.opt['model_path'].split('/')[-2]} model"
        )
        _, target_scores = self.target_model.get_scores(self.test_loader, False)
        target_labels = np.where(target_scores > self.target_threshold, 1.0, 0.0)

        _, p_target_scores = self.target_model.get_scores(self.pair_loader, False)
        p_target_labels = np.where(p_target_scores > self.target_threshold, 1.0, 0.0)

        ## protect
        self.logger.info(
            f"predict attr {self.opt['protected_attribute_name']} using {self.opt['model_path2'].split('/')[-2]} model"
        )
        _, pro_scores = self.protect_model.get_scores(self.test_loader, False)
        pro_labels = np.where(pro_scores > self.protect_threshold, 1.0, 0.0)

        _, p_pro_scores = self.protect_model.get_scores(self.pair_loader, False)
        p_pro_labels = np.where(p_pro_scores > self.protect_threshold, 1.0, 0.0)

        ### prepare imglist
        img_dir = self.data_setting['data_dir']
        img_pair_dir = self.data_setting['data_pair_dir']

        img_list, img_pair_list = [], []

        for img_id in range(0, len(target_labels)):
            img = self.opt['test_iname'] + '_' + str(img_id) + '.png'
            img_list.append(os.path.join(img_dir, img))
            img_pair_list.append(os.path.join(img_pair_dir, img))

        ## just consider target
        t_name = 'target_dis'
        with open(os.path.join(self.opt['expdir'], t_name + '.txt'), 'w') as f:
            dis_cnt = 0
            for ind in range(0, len(target_labels)):
                if target_labels[ind] != p_target_labels[ind]:
                    dis_cnt += 1
                    str_info = (
                        img_list[ind]
                        + ','
                        + img_pair_list[ind]
                        + ','
                        + str(target_labels[ind])
                        + ','
                        + str(p_target_labels[ind])
                    )
                    f.write(str_info + '\n')
            self.logger.info(f"target discrimination is {dis_cnt}")

        ## consider both target and protect attribute
        t_name = 'target_dis_pro'
        with open(os.path.join(self.opt['expdir'], t_name + '.txt'), 'w') as f:
            dis_cnt = 0
            for ind in range(0, len(target_labels)):
                if (
                    target_labels[ind] != p_target_labels[ind]
                    and pro_labels[ind] != p_pro_labels[ind]
                ):
                    dis_cnt += 1
                    str_info = str_info = (
                        img_list[ind]
                        + ','
                        + img_pair_list[ind]
                        + ','
                        + str(target_labels[ind])
                        + ','
                        + str(p_target_labels[ind])
                        + ','
                        + str(pro_labels[ind])
                        + ','
                        + str(p_pro_labels[ind])
                    )
                    f.write(str_info + '\n')
            self.logger.info(f"target_pro discrimination is {dis_cnt}")

        self.logger.info(f"total image is {len(target_labels)}")


if __name__ == '__main__':
    opt, logger = collect_args_fair_test()

    modelTest = ModelTest(opt, logger)
    start = time.time()
    modelTest.run_test()
    end = time.time()
    logger.info(f"test time is {end-start}")
