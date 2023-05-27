"""
train_kfold_cv.py

This module contains the implementation of the main training loop.
"""


import argparse

from models.main_painAttnNet import PainAttnNet
from parser import ConfigParser
from trainers.main_trainer import Trainer
from utils.utils import *

import torch
import torch.nn as nn

# Fix random seeds for reproducibility
SEED = 5272023
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


def weights_init_normal(m):
    """
    Initial weights
    """
    # torch.nn.init.xavier_normal_(m.weight.data)
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train_kfold(config, fold_id):
    batch_size = config["data_loader"]["args"]["batch_size"]

    logger = config.get_logger('trainers')

    # Build model architecture, initialize weights, then print to console
    model = PainAttnNet()

    model.apply(weights_init_normal)

    logger.info(model)


    # Get optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params)

    data_loader, valid_data_loader, data_count = load_data(folds_data[fold_id][0], folds_data[fold_id][1], batch_size)
    # weights_for_each_class = calc_class_weight(data_count)
    # weights_for_each_class = [1, 1, 1, 1, 1]

    device = torch.device('cuda:0' if config['n_gpu'] > 0 else 'cpu')
    # loss = nn.CrossEntropyLoss(weight=torch.tensor(weights_for_each_class).to(device))
    loss = nn.CrossEntropyLoss()

    trainer = Trainer(model, loss, optimizer,
                      config=config,
                      data_loader=data_loader,
                      fold_id=fold_id,
                      valid_data_loader=valid_data_loader)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for training arguments')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='config file path (default: config.json')
    parser.add_argument('-d', '--device', default='0', type=str,
                        help='GPUs')
    parser.add_argument('-f', '--fold_id', default = 0, type=int,
                        help='fold_id')
    parser.add_argument('-da', '--np_data_dir', default = r'Z:\Pison\pison_movement\data\processed', type=str,
                        help='Directory containing numpy files')

    args = parser.parse_args()
    config = ConfigParser.from_args(parser, args.fold_id)
    folds_data = generate_kfolds_index(args.np_data_dir, config["data_loader"]["args"]["num_folds"])

    train_kfold(config, args.fold_id)

