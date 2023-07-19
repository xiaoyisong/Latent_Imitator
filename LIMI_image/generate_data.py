from __future__ import absolute_import
import logging
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import cv2
import argparse
from utils import util
from utils.logger import setup_logger
import pickle
from Models.attr_classifier import attribute_classifier
import torchvision.transforms as T
from PIL import Image
from os import path
from tqdm import tqdm

IMG_DIR = './exp/img'


def _collect_args_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['orig'], default='orig')
    parser.add_argument('--exp_name', type=str, default='_')
    parser.add_argument('--attribute', type=int, default=31)
    parser.add_argument('--save_dir', type=str, default='_')
    parser.add_argument('--latent_file', type=str, default='_')
    parser.add_argument('--random_seed', type=int, default=2333)
    parser.add_argument('--num_images', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--protected_attribute', type=int, default=20)
    parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())

    attr_list = util.get_all_attr()
    opt['attr_name'] = attr_list[opt['attribute']]
    opt['prot_attr_name'] = attr_list[opt['protected_attribute']]
    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
    opt['dtype'] = torch.float32
    opt['expdir'] = os.path.join(IMG_DIR, opt['exp_name'])
    util.make_dir(opt['expdir'])

    if opt['experiment'] == 'orig' and opt['save_dir'] == '_':
        opt['save_dir'] = os.path.join(opt['expdir'], 'imgs')
        util.make_dir(opt['save_dir'])

    if opt['experiment'] == 'orig' and opt['latent_file'] == '_':
        opt['latent_file'] = os.path.join(opt['expdir'], 'latent_vectors_origin.pkl')

    if opt['expdir'] not in opt['save_dir']:
        opt['save_dir'] = os.path.join(opt['expdir'], opt['save_dir'])

    util.make_dir(opt['save_dir'])

    mode = 'w'
    logger = setup_logger(opt['expdir'], logger_name='logger', mode=mode)

    # Uncomment if deterministic run required
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opt['random_seed'])
    np.random.seed(opt['random_seed'])
    logger.info(f"the random_seed is set as {opt['random_seed']}\n")
    logger.info(f"setting {opt}\n")

    return opt


def generate_orig_images(model, opt):

    num_images = opt['num_images']
    batch_size = opt['batch_size']
    noise, _ = model.buildNoiseData(num_images)

    logger.info(f"Generating new images. Latent vectors stored at {opt['latent_file']}")
    # Saving latent vectors

    with open(opt['latent_file'], 'wb+') as f:
        pickle.dump(noise.detach().cpu().numpy(), f)

    N = int(num_images / batch_size)
    if num_images % batch_size != 0:
        N += 1
    count = 0
    for ind in tqdm(range(N)):
        with torch.no_grad():
            generated_images = model.test(
                noise[ind * batch_size : (ind + 1) * batch_size]
            )

        for i in range(generated_images.shape[0]):
            grid = torchvision.utils.save_image(
                generated_images[i].clamp(min=-1, max=1),
                os.path.join(opt['save_dir'], 'gen_' + str(count) + '.png'),
                padding=0,
                scale_each=True,
                normalize=True,
            )
            count += 1

    logger.info(f'{count} images are generated')


if __name__ == "__main__":

    opt = _collect_args_generate()
    logger = logging.getLogger('logger')

    use_gpu = True if torch.cuda.is_available() else False

    model = torch.hub.load(
        'facebookresearch/pytorch_GAN_zoo:hub',
        'PGAN',
        model_name='celebAHQ-256',
        pretrained=True,
        useGPU=use_gpu,
    )
    logger.info(f"load model finish")

    if opt['experiment'] == 'orig':
        generate_orig_images(model, opt)
