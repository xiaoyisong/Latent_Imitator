import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, recall_score
from os import listdir, path, mkdir
from scipy.ndimage import gaussian_filter


def get_all_attr():

    return [
        '5_o_Clock_Shadow',
        'Arched_Eyebrows',
        'Attractive',
        'Bags_Under_Eyes',
        'Bald',
        'Bangs',
        'Big_Lips',
        'Big_Nose',
        'Black_Hair',
        'Blond_Hair',
        'Blurry',
        'Brown_Hair',
        'Bushy_Eyebrows',
        'Chubby',
        'Double_Chin',
        'Eyeglasses',
        'Goatee',
        'Gray_Hair',
        'Heavy_Makeup',
        'High_Cheekbones',
        'Male',
        'Mouth_Slightly_Open',
        'Mustache',
        'Narrow_Eyes',
        'No_Beard',
        'Oval_Face',
        'Pale_Skin',
        'Pointy_Nose',
        'Receding_Hairline',
        'Rosy_Cheeks',
        'Sideburns',
        'Smiling',
        'Straight_Hair',
        'Wavy_Hair',
        'Wearing_Earrings',
        'Wearing_Hat',
        'Wearing_Lipstick',
        'Wearing_Necklace',
        'Wearing_Necktie',
        'Young',
    ]


def get_attr_list():
    return [
        1,
        2,
        3,
        5,
        6,
        7,
        8,
        9,
        11,
        12,
        13,
        15,
        17,
        19,
        21,
        23,
        25,
        26,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        39,
    ]


def make_dir(pathname):
    if not path.isdir(pathname):
        os.makedirs(pathname, exist_ok=True)


def get_threshold(targets_all, scores_all):
    best_t = -1.0
    best_acc = 0.0
    for t in range(1, 10):
        thresh = 0.1 * t
        curr_scores = np.where(scores_all > thresh, 1, 0)
        acc = f1_score(targets_all, curr_scores)
        # print(thresh, acc, best_acc, flush=True)
        if acc > best_acc:
            best_acc = acc
            best_t = thresh
    one_dec = best_t

    for t in range(1, 20):
        thresh = (one_dec - 0.1) + 0.01 * t
        curr_scores = np.where(scores_all > thresh, 1, 0)
        acc = f1_score(targets_all, curr_scores)
        # print(thresh, acc, best_acc, flush=True)
        if acc > best_acc:
            best_acc = acc
            best_t = thresh

    return best_acc, best_t


def calibrated_threshold(targets, scores):
    cp = int(targets.sum())
    scores_copy = np.copy(scores)
    scores_copy.sort()
    # print(cp)
    thresh = scores_copy[-cp]
    return thresh
