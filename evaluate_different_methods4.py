import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score

import torch
from torch_geometric.datasets import Planetoid, Amazon
from ogb.nodeproppred import PygNodePropPredDataset


def get_ytest(dataset, noise_type, mislabel_rate, target_set):
    # get y_test ( 1 indicates noisy / wrong label)
    if dataset == 'Flickr':
        class_map = './dataset/Flickr/raw/class_map.json'
        noisy_class_map = './dataset/Flickr/raw/' + 'noisy_class_map_' + noise_type + '_' + \
                          str(mislabel_rate) + '.json'
        role = './dataset/Flickr/raw/role.json'
        with open(class_map, 'r') as f:
            class_map = json.load(f)
        with open(noisy_class_map, 'r') as f:
            noisy_class_map = json.load(f)
        with open(role, 'r') as f:
            role = json.load(f)
        tr_mask = np.zeros(len(class_map), dtype=bool)
        if target_set == 'valid':
            tr_mask[role['va']] = True
        elif target_set == 'train':
            tr_mask[role['tr']] = True
        else:
            tr_mask[role['te']] = True
        origin_class = np.array(list(class_map.values()))[tr_mask]
        noisy_class = np.array(list(noisy_class_map.values()))[tr_mask]
        y_test = origin_class != noisy_class
    elif dataset in ['Cora', 'CiteSeer', 'PubMed']:
        data = Planetoid(root='./dataset', name=dataset)
        data = data[0]
        noisy_class_map = './dataset/' + dataset + '/raw/' + 'noisy_class_map_' + noise_type + '_' + \
                          str(mislabel_rate) + '.json'
        with open(noisy_class_map, 'r') as f:
            noisy_class_map = json.load(f)
        if target_set == 'valid':
            mask = data.val_mask
        elif target_set == 'train':
            mask = data.train_mask
        else:
            mask = data.test_mask
        origin_class = np.array(list(data.y))[mask]
        noisy_class = np.array(list(noisy_class_map.values()))[mask]
        y_test = origin_class != noisy_class
    elif dataset in ['Computers', 'Photo']:
        data = Amazon(root='./dataset/Amazon', name=dataset)
        data = data[0]
        length = len(data.y)
        noisy_class_map = './dataset/Amazon/' + dataset + '/raw/' + 'noisy_class_map_' + noise_type + '_' + \
                          str(mislabel_rate) + '.json'
        with open(noisy_class_map, 'r') as f:
            noisy_class_map = json.load(f)
        if target_set == 'valid':
            val_mask = np.zeros(length)
            val_mask[int(0.6 * length):int(0.8 * length)] = 1
            data.val_mask = torch.from_numpy(val_mask).bool()
            mask = data.val_mask
        elif target_set == 'train':
            train_mask = np.ones(length)
            train_mask[int(0.6 * length):] = 0
            data.train_mask = torch.from_numpy(train_mask).bool()
            mask = data.train_mask
        else:
            test_mask = np.zeros(length)
            test_mask[int(0.8 * length):] = 1
            data.test_mask = torch.from_numpy(test_mask).bool()
            mask = data.test_mask
        origin_class = np.array(list(data.y))[mask]
        noisy_class = np.array(list(noisy_class_map.values()))[mask]
        y_test = origin_class != noisy_class
    elif dataset in ['ogbn-arxiv']:
        noisy_class_map = '/data/yuwen/' + dataset.replace('-', '_') + '/raw/' + 'noisy_class_map_' + noise_type + '_' + \
                          str(mislabel_rate) + '.json'
        dataset = PygNodePropPredDataset(name=dataset, root='./dataset')
        split_idx = dataset.get_idx_split()
        mask = split_idx[target_set]
        data = dataset[0]
        with open(noisy_class_map, 'r') as f:
            noisy_class_map = json.load(f)
        origin_class = np.array(list(data.y.squeeze()))[mask]
        noisy_class = np.array(list(noisy_class_map.values()))[mask]
        y_test = origin_class != noisy_class
    return y_test


def cal_afpr(y_pred, y_test):
    print('Accuracy: {:.4f}; F1 Score: {:.4f}; Precision: {:.4f}; Recall: {:.4f}'.format
          (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred),
           precision_score(y_test, y_pred), recall_score(y_test, y_pred)))
    return f1_score(y_test, y_pred)


def cal_ap(ordered_idx, y_test):
    true_cnt = 0
    ap = 0
    for i in range(len(ordered_idx)):
        if np.isnan(ordered_idx[i]):
            break
        if y_test[int(ordered_idx[i])]:
            true_cnt += 1
            ap += true_cnt / (i+1)
    ap = ap / np.sum(y_test)
    print("Average Precision: {:.4f}".format(ap))
    return ap


def cal_mcc(y_pred, y_test):
    print('MCC: {:.4f}'.format(matthews_corrcoef(y_test, y_pred)))
    return matthews_corrcoef(y_test, y_pred)


def cal_patk(ordered_idx, y_test):
    true_cnt = 0
    for i in range(np.sum(y_test)):
        if np.isnan(ordered_idx[i]):
            break
        if y_test[int(ordered_idx[i])]:
            true_cnt += 1
    ap = true_cnt / np.sum(y_test)
    print("Precision @ |true|: {:.4f}".format(ap))
    return ap


def cal_auc(y_test, score):
    roc_auc = roc_auc_score(y_test, score)
    print("ROC AUC Score: {:.4f}".format(roc_auc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating Different Methods")
    parser.add_argument("--exp", type=int, default=0, help="-1 means evaluate 0-9 runs")
    parser.add_argument("--dataset", type=str, default='Flickr')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--mislabel_rate", type=float, default=0.1)
    parser.add_argument("--sample_rate", type=float, default=0.5)
    parser.add_argument("--noise_type", type=str, default='symmetric')
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--classifier", type=str, default='MLP')
    parser.add_argument("--method", type=str, default='CL',
                        help='If want to evaluate more than 1 method, link method names with \'+\'.')
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--validation', type=bool, default=True)
    parser.add_argument("--test_target", type=str, default='test')
    args = parser.parse_args()

    # get y_test ( 1 indicates noisy / wrong label)
    # y_test = get_ytest(args.dataset, args.noise_type, args.mislabel_rate, args.test_target)

    ans = []
    for noise_type in ["symmetric", "asymmetric"]:
        for choice in ["noCL", "lbonly", "nbonly", "ori"]:
            tmp = []
            for dataset in ["Cora", "Computers", "ogbn-arxiv"]:
                y_test = get_ytest(dataset, noise_type, 0.1, args.test_target)
                F1, MCC, P = [], [], []
                for run in range(10):
                    if choice == "ori":
                        mislabel_result_file = 'mislabel_results/validl1-laplacian-test=test-MLP-{}-{}-mislabel={}-{}-sample=0.5-k={}-epochs=200-' \
                                               'lr=0.001-wd=0.0005-exp={}'.format(dataset, "GCN", 0.1, noise_type, args.k, run)
                    elif choice == "noCL":
                        mislabel_result_file = 'mislabel_results/validl1-noCL-laplacian-test=test-MLP-{}-{}-mislabel={}-{}-sample=0.5-k={}-epochs=200-' \
                                               'lr=0.001-wd=0.0005-exp={}'.format(dataset, "GCN", 0.1, noise_type,
                                                                                  args.k, run)
                    elif choice == "lbonly":
                        mislabel_result_file = 'mislabel_results/validl1-laplacian-test=test-MLP-{}-{}-mislabel={}-{}-sample=0.5-k={}-epochs=200-' \
                                               'lr=0.001-wd=0.0005-lbonly-exp={}'.format(dataset, "GCN", 0.1, noise_type,
                                                                                  args.k, run)
                    else:
                        mislabel_result_file = 'mislabel_results/validl1-laplacian-test=test-MLP-{}-{}-mislabel={}-{}-sample=0.5-k={}-epochs=200-' \
                                               'lr=0.001-wd=0.0005-nbonly-exp={}'.format(dataset, "GCN", 0.1, noise_type,
                                                                                  args.k, run)
                    mislabel_result_file += '.csv'
                    mislabel_result = pd.read_csv(mislabel_result_file)
                    print("Evaluate ", choice)
                    F1.append(cal_afpr(np.array(mislabel_result['result']), y_test))
                    MCC.append(cal_mcc(np.array(mislabel_result['result']), y_test))
                    P.append(cal_patk(np.array(mislabel_result['ordered_errors']), y_test))

                tmp.append(np.mean(F1))
                tmp.append(np.mean(MCC))
                tmp.append(np.mean(P))
            ans.append(tmp)

    print(ans)
