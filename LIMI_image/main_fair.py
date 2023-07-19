import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
import time
import argparse
from utils.logger import setup_logger
from utils import util

import torch
import torchvision

EXP_DIR = "./exp"


def collect_args_edit_latent():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='main_fair')
    parser.add_argument('--exp_name', type=str, default='_')
    parser.add_argument(
        '--exp_flag', choices=['generate', 'test'], default='generate',
    )
    parser.add_argument('--latent_file', type=str, default='_')
    parser.add_argument('--model_path', type=str, default='_')
    parser.add_argument('--num_images', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument(
        '--boundary_dir', type=str, default='./exp/train_boundaries',
    )

    parser.add_argument('--target_attribute', type=int, default=31)
    parser.add_argument('--protected_attribute', type=int, default=20)  # gender

    parser.add_argument('--random_seed', type=int, default=2333)
    parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())
    attr_list = util.get_all_attr()

    attr_name = attr_list[opt['protected_attribute']]
    opt['protected_attribute_name'] = attr_name

    attr_name = attr_list[opt['target_attribute']]
    opt['target_attribute_name'] = attr_name

    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
    opt['dtype'] = torch.float32

    if opt['exp_name'] == '_':
        opt['exp_name'] = (
            str(opt['target_attribute']) + '_' + opt['target_attribute_name']
        )

    opt['expdir'] = os.path.join(EXP_DIR, opt['experiment'], opt['exp_name'])
    util.make_dir(opt['expdir'])

    opt['save_dir_p1'] = os.path.join(opt['expdir'], 'imgs_p1')
    opt['save_dir_p2'] = os.path.join(opt['expdir'], 'imgs_p2')

    util.make_dir(opt['save_dir_p1'])
    util.make_dir(opt['save_dir_p2'])

    # deal the path of boundary
    path = os.path.join(
        opt['boundary_dir'],
        str(opt['target_attribute']) + '_' + opt['target_attribute_name'] + '.npy',
    )
    opt['target_boundary'] = path

    path = os.path.join(
        opt['boundary_dir'],
        str(opt['protected_attribute'])
        + '_'
        + opt['protected_attribute_name']
        + '.npy',
    )
    opt['protected_boundary'] = path

    if opt['exp_flag'] == 'generate':
        logger = setup_logger(
            opt['expdir'], logfile_name='generate.txt', logger_name='logger'
        )
    elif opt['exp_flag'] == 'test':
        logger = setup_logger(
            opt['expdir'], logfile_name='test.txt', logger_name='logger'
        )
    # Uncomment if deterministic run required
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opt['random_seed'])
    np.random.seed(opt['random_seed'])
    logger.info(f"the random_seed is set as {opt['random_seed']}\n")
    logger.info(f"setting {opt}\n")

    return opt, logger


class FairTest:
    def __init__(self, opt, logger) -> None:
        self.opt = opt
        self.logger = logger

    def build_data_and_gan(self):
        with open(self.opt['latent_file'], 'rb') as f:
            self.latent_codes = pickle.load(f)

        self.model = torch.hub.load(
            'facebookresearch/pytorch_GAN_zoo:hub',
            'PGAN',
            model_name='celebAHQ-256',
            pretrained=True,
            useGPU=True,
        )
        self.logger.info(f"load model finish")

        self.bd_target = self.load_boundary(self.opt['target_boundary'])
        self.bd_protected = self.load_boundary(opt['protected_boundary'])
        self.dis2target = self.calculate_dis(self.latent_codes, self.bd_target)

    def calculate_dis(self, latent_codes, boundary):
        dis_samples = []
        for index in range(0, len(latent_codes)):
            latent_sample = latent_codes[index : index + 1]
            dis_sample = (
                np.sum(boundary['coef_'] * latent_sample) + boundary['intercept_']
            )
            dis_samples.append(dis_sample[0])
        return np.array(dis_samples)

    def load_boundary(self, path: str):
        with open(path, 'rb') as f:
            bd = pickle.load(f)
        ans = bd
        return ans

    def generate_imgs(self, latents, save_dir, iname='gen'):
        noise = torch.Tensor(latents)
        batch_size = self.opt['batch_size']
        num_images = noise.shape[0]
        N = int(num_images / batch_size)

        if num_images % batch_size != 0:
            N += 1
        count = 0
        for ind in tqdm(range(N)):
            with torch.no_grad():
                generated_images = self.model.test(
                    noise[ind * batch_size : (ind + 1) * batch_size]
                )
            for i in range(generated_images.shape[0]):
                grid = torchvision.utils.save_image(
                    generated_images[i].clamp(min=-1, max=1),
                    os.path.join(save_dir, iname + '_' + str(count) + '.png'),
                    padding=0,
                    scale_each=True,
                    normalize=True,
                )
                count += 1

        self.logger.info(f'{count} images are generated')
        self.logger.info(f"images are saved in {save_dir}")

    def run_generate(self):
        zero_latent = np.zeros((self.latent_codes.shape[0], self.latent_codes.shape[1]))
        candidate_n1 = np.zeros(
            (self.latent_codes.shape[0], self.latent_codes.shape[1])
        )
        candidate_p1 = np.zeros(
            (self.latent_codes.shape[0], self.latent_codes.shape[1])
        )
        step = 0.3
        for ind in tqdm(range(len(self.latent_codes))):
            t_z = self.latent_codes[ind : ind + 1]
            t_dis = self.dis2target[ind]

            sample_zero = t_z - t_dis * self.bd_target['coef_']
            zero_latent[ind] = sample_zero
            # just perturb once
            latent_edit = sample_zero + step * self.bd_target['coef_']
            candidate_p1[ind] = latent_edit
            latent_edit = sample_zero - step * self.bd_target['coef_']
            candidate_n1[ind] = latent_edit

        ## perturb gender
        p_zero_latent = np.zeros(
            (self.latent_codes.shape[0], self.latent_codes.shape[1])
        )
        p_candidate_n1 = np.zeros(
            (self.latent_codes.shape[0], self.latent_codes.shape[1])
        )
        p_candidate_p1 = np.zeros(
            (self.latent_codes.shape[0], self.latent_codes.shape[1])
        )
        zero_dis2pro = self.calculate_dis(zero_latent, self.bd_protected)
        n1_dis2pro = self.calculate_dis(candidate_n1, self.bd_protected)
        p1_dis2pro = self.calculate_dis(candidate_p1, self.bd_protected)
        for ind in tqdm(range(len(self.latent_codes))):
            p_zero_latent[ind] = (
                zero_latent[ind] - 2 * zero_dis2pro[ind] * self.bd_protected['coef_']
            )
            p_candidate_p1[ind] = (
                candidate_p1[ind] - 2 * p1_dis2pro[ind] * self.bd_protected['coef_']
            )
            p_candidate_n1[ind] = (
                candidate_n1[ind] - 2 * n1_dis2pro[ind] * self.bd_protected['coef_']
            )

        ###  six latent vectors, three pairs
        # p1 zero_latent,candidate_p1,candidate_n1
        # self.opt['save_dir_p1']
        # p2 p_zero_latent,p_candidate_p1,p_candidate_n1
        # self.opt['save_dir_p2']

        cz0_dir = os.path.join(self.opt['save_dir_p1'], 'cz0')
        util.make_dir(cz0_dir)
        self.generate_imgs(latents=zero_latent, save_dir=cz0_dir, iname='cz0')

        cz0_dir = os.path.join(self.opt['save_dir_p2'], 'cz0')
        util.make_dir(cz0_dir)
        self.generate_imgs(latents=p_zero_latent, save_dir=cz0_dir, iname='cz0')

        cp1_dir = os.path.join(self.opt['save_dir_p1'], 'cp1')
        util.make_dir(cp1_dir)
        self.generate_imgs(latents=candidate_p1, save_dir=cp1_dir, iname='cp1')

        cp1_dir = os.path.join(self.opt['save_dir_p2'], 'cp1')
        util.make_dir(cp1_dir)
        self.generate_imgs(latents=p_candidate_p1, save_dir=cp1_dir, iname='cp1')

        cn1_dir = os.path.join(self.opt['save_dir_p1'], 'cn1')
        util.make_dir(cn1_dir)
        self.generate_imgs(latents=candidate_n1, save_dir=cn1_dir, iname='cn1')

        cn1_dir = os.path.join(self.opt['save_dir_p2'], 'cn1')
        util.make_dir(cn1_dir)
        self.generate_imgs(latents=p_candidate_n1, save_dir=cn1_dir, iname='cn1')

    def run(self):
        if self.opt['exp_flag'] == 'generate':
            self.build_data_and_gan()
            self.run_generate()
        elif self.opt['exp_flag'] == 'test':
            pass


if __name__ == '__main__':

    opt, logger = collect_args_edit_latent()

    fairtest = FairTest(opt, logger)
    start = time.time()
    fairtest.run()
    end = time.time()
    logger.info(f"Total time is {end-start}")
