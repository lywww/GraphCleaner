import os
import json
import random
import pandas as pd
import numpy as np

import torch
from torch_geometric.datasets import Planetoid, Amazon
from ogb.nodeproppred import PygNodePropPredDataset

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


def setup():
    # seed = 1119
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_data(dataset, noise_type='symmetric', mislabel_rate=0, special_set=None):
    dir = os.path.join('./dataset', dataset)
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        data = Planetoid(root='./dataset', name=dataset)
        n_classes = data.num_classes
        data = data[0]
        print("number of training samples in get_data: ", sum(data.train_mask))

    elif dataset in ['Computers', 'Photo']:
        dir = os.path.join('./dataset/Amazon', dataset)
        data = Amazon(root='./dataset/Amazon', name=dataset)
        n_classes = data.num_classes
        data = data[0]
        length = len(data.y)
        train_mask = np.ones(length)
        train_mask[int(0.6 * length):] = 0
        val_mask = np.zeros(length)
        val_mask[int(0.6 * length):int(0.8 * length)] = 1
        test_mask = np.zeros(length)
        test_mask[int(0.8 * length):] = 1
        data.train_mask = torch.from_numpy(train_mask).bool()
        data.val_mask = torch.from_numpy(val_mask).bool()
        data.test_mask = torch.from_numpy(test_mask).bool()
        print("number of training samples in get_data: ", sum(data.train_mask))

    elif dataset in ['ogbn-arxiv']:
        dir = dir.replace('-', '_')
        data = PygNodePropPredDataset(name=dataset, root='./dataset')
        split_idx = data.get_idx_split()
        n_classes = data.num_classes
        data = data[0]
        train_mask, val_mask, test_mask = torch.zeros(len(data.y)).bool(), torch.zeros(len(data.y)).bool(), torch.zeros(len(data.y)).bool()
        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.y = data.y.squeeze()

    if mislabel_rate == 0:
        return data, n_classes

    with open(os.path.join(dir, 'raw', 'noisy_class_map_' + noise_type + '_' + str(mislabel_rate) + '.json'), 'r') as f:
        noisy_class_map = json.load(f)
    for i in range(len(noisy_class_map)):
        try:
            data.y[i] = noisy_class_map[str(i)]
        except:
            data.y[i] = noisy_class_map[i]
    if special_set and 'AUM' in special_set:
        threshold_samples = pd.read_csv('./aum_data/' + dataset + '_aum_threshold_samples.csv')
        if special_set[-1] == '1':
            threshold_idx = threshold_samples['first_threshold_samples']
        else:
            threshold_idx = threshold_samples['second_threshold_samples']
        for idx in threshold_idx:
            data.y[idx] = n_classes
        n_classes += 1

    return data, n_classes


def create_summary_writer(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def to_softmax(x):
    # transfer log_softmax to softmax
    max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp((x - max).astype(float))
    sum = np.sum(e_x, axis=1, keepdims=True)
    f_x = e_x / sum
    return f_x
