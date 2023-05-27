import glob
import os

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset, DataLoader

import math
from pathlib import Path
import json
from collections import OrderedDict
from itertools import repeat
import pandas as pd

def generate_kfolds_index(npz_dir, k_folds) -> dict[int: list[str]]:
    """
    Generate k-folds dataset index and store into a dictionary. The length of dictionary is equal to the number of
    folds. Each element contains training set and testing set.
    :param npz_dir: npz files directory
    :param k_folds: the number of folds
    :return: a dict contains k-folds dataset paths, e.g. dict{0: [list[str(train_dir)], list[str(test_dir)]]..., k:[...]}
    """

    if os.path.exists(npz_dir):
        print('================= Creating KFolds Index =================')
    else:
        print('================= Data directory does not exist =================')

    npz_files = glob.glob(os.path.join(npz_dir, '*.npz'))
    npz_files = np.asarray(npz_files)
    kfolds_names = np.array_split(npz_files, k_folds)
    # print(kfolds_names)
    kfolds_index = {}
    for fold_index in range(0, k_folds):
        test_data = kfolds_names[fold_index].tolist()
        train_data = [files for i, files in enumerate(kfolds_names) if i != fold_index]
        train_data = [files for subfiles in train_data for files in subfiles]
        kfolds_index[fold_index] = [train_data, test_data]
    print('================= {} folds dataset created ================='.format(k_folds))
    return kfolds_index


class PisonWristLoader(Dataset):
    """
    Input: a list of npz files' directories from k-folds index
    Output: a tensor of values and labels
    """
    def __init__(self, npz_files):
        super(PisonWristLoader, self).__init__()

        # Load first npz file which is easy to handle for the rest
        x_values = np.load(npz_files[0])['x']
        y_labels = np.load(npz_files[0])['y']

        # Load npz files starting from position 1
        for file in npz_files[1:]:
            x_values = np.vstack((x_values, np.load(file)['x']))
            y_labels = np.append(y_labels, np.load(file)['y'])
        # Change shape to (Batch size, Channel size, Length)
        x_values = np.transpose(x_values, (0, 2, 1))
        self.val = torch.from_numpy(x_values).float()
        self.lbl = torch.from_numpy(y_labels).long()

    def __len__(self):
        return self.val.shape[0]

    def __getitem__(self, idx):
        return self.val[idx], self.lbl[idx]

    def __repr__(self):
        return '{}'.format(repr(self.val))

    def __str__(self):
        # Pison total: torch.Size([15, 14, 967]), torch.Size([15])
        # If truncating then 967, if padding 1014.
        return 'The shape of values and labels: {}, {}'.format(self.val.shape, self.lbl.shape)


def load_data(train_set, valid_set, batch_size, num_workers = 0) -> tuple[DataLoader, DataLoader, list[int]]:
    """
    generate dataloader for both training dataset and validation dataset from one of the k-folds.
    :param train_set: training dataset
    :param valid_set: validation dataset
    :param batch_size: batch size
    :param num_workers: 4*GPU
    :return: dataloader for training dataset, validation dataset, the number of samples for each class,
        e.g. two classes -> list[int,int]
    """
    train_dataset = PisonWristLoader(train_set)
    valid_dataset = PisonWristLoader(valid_set)

    cat_y = torch.cat((train_dataset.lbl, valid_dataset.lbl))
    # dist = [cat_y.count(i) for i in range(n_classes)]

    # e.g. two classes (tensor(list[lbl, lbl]), tensor(list[int, int]))
    unique_counts = cat_y.unique(return_counts = True)
    # number of samples for each class -> list[int, int]
    dist = unique_counts[1].tolist()

    train_loader = DataLoader(train_dataset,
                              num_workers = num_workers,
                              batch_size = batch_size,
                              shuffle = True,
                              drop_last = False,
                              pin_memory = True)

    valid_loader = DataLoader(valid_dataset,
                              num_workers = num_workers,
                              batch_size = batch_size,
                              shuffle = False,
                              drop_last = False,
                              pin_memory = True)
    return train_loader, valid_loader, dist


def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight


def get_data_dir(directory):
    """
    This function returns the data directory.
    Generally, the data folder is supposed to put in the same level of src.

    Parameters:
    directory (str): The data folder name.

    Returns:
    str: The data directory.
    """
    data_dir = directory
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)

    return os.path.join(grandparent_dir, directory)

