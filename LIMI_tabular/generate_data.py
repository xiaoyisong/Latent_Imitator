import argparse
import logging
import os
import pickle
import torch
import numpy as np
from utils.logger import setup_logger
from ctgan.synthesizers.ctgan import CTGANSynthesizer
from utils import utils_gan

EXP_DIR = "./exp"
TABLE_DIR = "./exp/table"


def _parse_args():
    parser = argparse.ArgumentParser(description="generate data")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Batch size. Must be an even number.",
    )
    parser.add_argument(
        "--load_path",
        default=None,
        type=str,
        help="A filename to load a trained synthesizer.",
    )
    parser.add_argument("--exp_name", type=str, default="_")
    parser.add_argument("--exp_flag", type=str, default="_")
    parser.add_argument("--save_file", type=str, default="_")
    parser.add_argument("--latent_file", type=str, default="_")
    parser.add_argument("--random_seed", type=int, default=2333)
    parser.add_argument("--num_samples", type=int, default=100)
    opt = vars(parser.parse_args())

    opt["expdir"] = os.path.join(TABLE_DIR, opt["exp_name"])
    utils_gan.make_dir(opt["expdir"])

    if opt["save_file"] == "_":
        opt["save_file"] = "sampled_data.csv"
    if opt["expdir"] not in opt["save_file"]:
        opt["save_file"] = os.path.join(opt["expdir"], opt["save_file"])

    if opt["latent_file"] == "_":
        opt["latent_file"] = "sampled_latent.pkl"

    if opt['exp_flag'] == 'orig' and opt["expdir"] not in opt["latent_file"]:
        opt["latent_file"] = os.path.join(opt["expdir"], opt["latent_file"])

    mode = "a+"
    logger = setup_logger(opt["expdir"], logger_name="logger", mode=mode)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opt["random_seed"])
    np.random.seed(opt["random_seed"])
    logger.info(f"the random_seed is set as {opt['random_seed']}\n")
    logger.info(f"setting {opt}\n")

    return opt


def generate_orig(model: CTGANSynthesizer, opt):
    num_images = opt["num_samples"]
    batch_size = opt["batch_size"]

    logger.info(f"Generating new images. Latent vectors stored at {opt['latent_file']}")
    sampled, fake_data = model.sample(n=opt["num_samples"], need_fakez=True)
    with open(opt["latent_file"], "wb+") as handle:
        pickle.dump(fake_data, handle)
    sampled.to_csv(opt["save_file"], index=False)




if __name__ == "__main__":
    opt = _parse_args()
    logger = logging.getLogger("logger")
    model = CTGANSynthesizer.load(opt["load_path"])
    model.set_mode()
    logger.info(f"load model finish")
    if opt['exp_flag'] == 'orig':
        generate_orig(model=model, opt=opt)
