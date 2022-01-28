import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score

from torch_geometric.datasets import Planetoid


def get_ytest(dataset, noise_type, mislabel_rate, validation):
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
        if validation:
            tr_mask[role['va']] = True
        else:
            tr_mask[role['tr']] = True
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
        if validation:
            mask = data.val_mask
        else:
            mask = data.train_mask
        origin_class = np.array(list(data.y))[mask]
        noisy_class = np.array(list(noisy_class_map.values()))[mask]
        y_test = origin_class != noisy_class
    return y_test


def cal_afpr(y_pred, y_test):
    print('Accuracy: {:.2f}; F1 Score: {:.2f}; Precision: {:.2f}; Recall: {:.2f}'.format
          (accuracy_score(y_test, y_pred), f1_score(y_test, y_pred),
           precision_score(y_test, y_pred), recall_score(y_test, y_pred)))


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
    print("Average Precision: {:.2f}".format(ap))


def cal_mcc(y_pred, y_test):
    print('MCC: {:.2f}'.format(matthews_corrcoef(y_test, y_pred)))


def cal_patk(ordered_idx, y_test):
    true_cnt = 0
    for i in range(np.sum(y_test)):
        if np.isnan(ordered_idx[i]):
            break
        if y_test[int(ordered_idx[i])]:
            true_cnt += 1
    ap = true_cnt / np.sum(y_test)
    print("Precision @ |true|: {:.2f}".format(ap))


def cal_auc(y_test, score):
    roc_auc = roc_auc_score(y_test, score)
    print("ROC AUC Score: {:.2f}".format(roc_auc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating Different Methods")
    parser.add_argument("--dataset", type=str, default='Flickr')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--mislabel_rate", type=float, default=0.1)
    parser.add_argument("--sample_rate", type=float, default=0.5)
    parser.add_argument("--noise_type", type=str, default='symmetric')
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--classifier", type=str, default='LR')
    parser.add_argument("--method", type=str, default='CL',
                        help='If want to evaluate more than 1 method, link method names with \'+\'.')
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--validation', type=bool, default=True)
    args = parser.parse_args()

    # get y_test ( 1 indicates noisy / wrong label)
    y_test = get_ytest(args.dataset, args.noise_type, args.mislabel_rate, args.validation)

    for m in args.method.split('+'):
        if m == '':
            break

        # if m == 'CL':
        #     mislabel_result_file = 'mislabel_results/{}-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format \
        #         (m, args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
        #     mislabel_result_file += '.csv'
        #     mislabel_result = pd.read_csv(mislabel_result_file)
        #     print("Evaluate baseline_conf_joint_only...")
        #     cal_afpr(np.array(mislabel_result['baseline_conf_joint_only']), y_test)
        #     print("Evaluate baseline_argmax...")
        #     cal_afpr(np.array(mislabel_result['baseline_argmax']), y_test)
        #     print("Evaluate baseline_cl_pbc...")
        #     cal_afpr(np.array(mislabel_result['baseline_cl_pbc']), y_test)
        #     print("Evaluate baseline_cl_pbnr...")
        #     cal_afpr(np.array(mislabel_result['baseline_cl_pbnr']), y_test)
        #     print("Evaluate baseline_cl_both...")
        #     cal_afpr(np.array(mislabel_result['baseline_cl_both']), y_test)
        # elif m == 'baseline' or m == 'AUM':
        if args.validation:
            mislabel_result_file = 'mislabel_results/validation-{}-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format \
            (m, args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr,
             args.weight_decay)
        else:
            mislabel_result_file = 'mislabel_results/{}-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format \
                (m, args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr,
                 args.weight_decay)
        if m == 'ours':
            mislabel_result_file = 'mislabel_results/{}-{}-{}-mislabel={}-{}-sample={}-k={}-epochs={}-lr={}-wd={}'.\
                format(args.classifier, args.dataset, args.model, args.mislabel_rate, args.noise_type, args.sample_rate,
                       args.k, args.n_epochs, args.lr, args.weight_decay)
        mislabel_result_file += '.csv'
        mislabel_result = pd.read_csv(mislabel_result_file)
        print("Evaluate ", m)
        cal_afpr(np.array(mislabel_result['result']), y_test)
        cal_mcc(np.array(mislabel_result['result']), y_test)
        cal_patk(np.array(mislabel_result['ordered_errors']), y_test)

        if m in ['DYB', 'ours']:
            cal_auc(y_test, np.array(mislabel_result['score']))
        if m in ['AUM']:
            cal_auc(y_test, -np.array(mislabel_result['score']))
