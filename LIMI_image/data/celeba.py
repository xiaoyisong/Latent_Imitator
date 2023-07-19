from __future__ import print_function, division
import logging

import os
import glob
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchvision.transforms as T
import pickle
import itertools

logger = logging.getLogger('logger')


class CelebaDataset(Dataset):
    def __init__(self, list_IDs, labels, augment):
        self.labels = labels
        self.list_IDs = list_IDs
        self.augment = augment
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if augment:
            self.transform = T.Compose(
                [
                    # T.Resize(64),
                    # T.Resize(256),
                    # T.RandomCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    # T.Resize(64),
                    # T.Resize(256),
                    # T.CenterCrop(224),
                    T.ToTensor(),
                    normalize,
                ]
            )

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        img = Image.open(ID)

        X = self.transform(img)
        y = self.labels[ID]

        return X, y


def create_dataset_celeba(
    data_dir, attr_file, attribute, protected_attribute, params, augment, split='train'
):

    list_ids = []
    with open(attr_file, 'r') as f:
        labels = f.readlines()

    train_index = 0
    valid_index = 162770
    test_index = 182637

    begin_index, end_index = 0, 0
    if split == 'train':
        begin_index, end_index = train_index, valid_index
    elif split == 'val':
        begin_index, end_index = valid_index, test_index
    elif split == 'test':
        begin_index, end_index = test_index, 202599 - test_index
    else:

        logger.debug(f'{split} Error')
        return
    attr = {}
    for i in range(begin_index + 2, end_index + 2):
        temp = labels[i].strip().split()
        list_ids.append(os.path.join(data_dir, temp[0]))
        attr[os.path.join(data_dir, temp[0])] = torch.Tensor(
            [
                int((int(temp[attribute + 1]) + 1) / 2),
                int((int(temp[protected_attribute + 1]) + 1) / 2),
            ]
        )

    dataset = CelebaDataset(list_ids, attr, augment)
    dataloader = DataLoader(dataset, **params)

    return dataloader


def create_dataset_synthesis(data_dir, params, augment, number=None, iname='cz0'):

    list_ids = []
    attr = {}
    if number == None:
        number = len(os.listdir(data_dir))

    print(number)
    for img_id in range(0, number):
        img = iname + '_' + str(img_id) + '.png'
        list_ids.append(os.path.join(data_dir, img))
        attr[os.path.join(data_dir, img)] = -1

    dataset = CelebaDataset(list_ids, attr, augment)
    dataloader = DataLoader(dataset, **params)
    logger.info(f' len of dataset {len(dataset)}')
    return dataloader
