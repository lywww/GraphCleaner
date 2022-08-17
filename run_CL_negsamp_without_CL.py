import os
import copy
from tqdm import tqdm
import random
import argparse
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.sparse import spdiags

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn.models import MLP

from GNN_models import GCN, myGIN, GAT, baseMLP
from Utils import to_softmax, get_data, create_summary_writer, ensure_dir, setup
from evaluate_different_methods import cal_patk, cal_afpr, get_ytest, cal_mcc


def train_GNNs(model_name, dataset, n_epochs, lr, wd, trained_model_file, mislabel_rate, noise_type, batch_size, neg_data=None):
    # prepare data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    data, n_classes = get_data(dataset, noise_type, mislabel_rate)
    if neg_data:
        data = neg_data
    n_features = data.num_features
    print("Data: ", data)

    # prepare model
    if model_name == 'GCN':
        model = GCN(in_channels=n_features, hidden_channels=256, out_channels=n_classes)
    elif model_name == 'GIN':
        model = myGIN(in_channels=n_features, hidden_channels=256, out_channels=n_classes)
    elif model_name == 'GAT':
        model = GAT(in_channels=n_features, hidden_channels=256, out_channels=n_classes)
    elif model_name == 'MLP':
        model = baseMLP([n_features, 256, n_classes])
    model.to(device)
    print("Model: ", model)

    # prepare optimizer and dataloader
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # load / train the model
    all_sm_vectors = []
    all_two_logits = []
    best_sm_vectors = []
    best_cri = 0
    if neg_data:
        model.load_state_dict(torch.load(trained_model_file))
    for epoch in range(n_epochs):
        model.train()
        data.to(device)
        optimizer.zero_grad()
        out = model(data)
        if model_name == 'MLP':
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        else:
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        print("Epoch[{}] Loss: {:.2f}".format(epoch + 1, loss))
        loss.backward()
        optimizer.step()

        model.eval()
        eval_out = model(data)
        y_pred = eval_out[data.val_mask].argmax(dim=-1).cpu().detach()
        y_true = data.y[data.val_mask].cpu().detach()
        cri = f1_score(y_true, y_pred, average='micro')
        if cri > best_cri:
            print("New Best Criterion: {:.2f}".format(cri))
            best_cri = cri
            best_sm_vectors = eval_out.cpu().detach().numpy()
            torch.save(model.state_dict(), trained_model_file)

        if neg_data:
            if (epoch + 1) % 1 == 0:
                all_sm_vectors.append(F.nll_loss(out, data.y, reduction='none').cpu().detach().numpy())
        else:
            if (epoch+1) % 20 == 0:
                all_sm_vectors.append(eval_out.cpu().detach().numpy())

    return np.array(all_sm_vectors), np.array(all_two_logits), to_softmax(best_sm_vectors), data.cpu().detach(), n_classes


def negative_sampling(data_orig, sample_rate, n_classes):
    data = copy.deepcopy(data_orig)
    train_idx = np.argwhere(data.held_mask == True)[0]

    # sampling and change classes
    tr_sample_idx = random.sample(list(train_idx), int(np.round(sample_rate * len(train_idx))))
    for idx in tr_sample_idx:
        y = int(data.y[idx])
        while y == int(data.y[idx]):
            y = np.random.choice(n_classes)
        data.y[idx] = y
    return data, tr_sample_idx


def generate_new_feature(k, data, noisy_data, sample_idx, all_sm_vectors, all_two_logits, n_classes):
    print("Generating new features......")

    y = np.zeros(data.num_nodes)  # 1 indicates negative / noisy
    y[sample_idx] = 1

    A = torch_geometric.utils.convert.to_scipy_sparse_matrix(data.edge_index)
    n = A.shape[0]

    # symmetric normalized adjacency matrix. 1e-10 is to prevent dividing by zero.
    S = spdiags(np.squeeze((1e-10 + np.array(A.sum(1)))**-0.5), 0, n, n) @ A @ spdiags(np.squeeze((1e-10 + np.array(A.sum(0)))**-0.5), 0, n, n)
    S2 = S @ S
    S3 = S @ S2
    S2.setdiag(np.zeros((n,)))  # remove self-loops to prevent the algorithm peeking at the original label (L0)
    S3.setdiag(np.zeros((n,)))
    print("S calculated!")

    ymat = y[:, np.newaxis]
    L0 = np.eye(n_classes)[data.y]  # original labels, converted to one-hot matrix
    L_corr = (ymat * np.eye(n_classes)[noisy_data.y]) + (1 - ymat) * L0  # label matrix corrupted by negative samples
    L1 = S @ L0
    L2 = S2 @ L0
    L3 = S3 @ L0
    print("L calculated!")

    P0 = scipy.special.softmax(all_sm_vectors[-1, :, :], axis=1)  # base model softmax predictions matrix
    P1 = S @ P0
    P2 = S2 @ P0
    P3 = S3 @ P0
    print("P calculated!")

    feat = np.hstack((
        np.sum(L_corr * P0, axis=1, keepdims=True),  # since L_corr is one-hot, this just extracts the corresponding entry of P0
        np.sum(L_corr * P1, axis=1, keepdims=True),
        np.sum(L_corr * P2, axis=1, keepdims=True),
        np.sum(L_corr * P3, axis=1, keepdims=True),
        np.sum(L_corr * L1, axis=1, keepdims=True),
        np.sum(L_corr * L2, axis=1, keepdims=True),
        np.sum(L_corr * L3, axis=1, keepdims=True),
    ))
    return feat, y


class myMLP(torch.nn.Module):
    def __init__(self, channel_list, dropout=0.0, relu_first=True):
        super().__init__()
        self.mlp = MLP(channel_list=channel_list, dropout=dropout, relu_first=relu_first, batch_norm=True)

    def forward(self, data):
        f = self.mlp(data)
        y = F.softmax(f, dim=1)
        return y


if __name__ == "__main__":
    setup()

    parser = argparse.ArgumentParser(description="GraphCleaner (no CL)")
    parser.add_argument("--exp", type=int, default=0)
    parser.add_argument("--dataset", type=str, default='Cora')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--mislabel_rate", type=float, default=0.1)
    parser.add_argument("--noise_type", type=str, default='symmetric')
    parser.add_argument("--sample_rate", type=float, default=0.5)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--classifier", type=str, default='MLP')
    parser.add_argument("--held_split", type=str, default='valid')
    parser.add_argument("--test_target", type=str, default='test')
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--validation', type=bool, default=True)
    args = parser.parse_args()

    ensure_dir('checkpoints')
    trained_model_file = 'checkpoints/{}-{}-mislabel={}-{}-epochs={}-bs={}-lr={}-wd={}-exp={}'.format\
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.batch_size, args.lr,
         args.weight_decay, args.exp)
    ensure_dir('mislabel_results')
    mislabel_result_file = 'mislabel_results/validl1-noCL-laplacian-test={}-{}-{}-{}-mislabel={}-{}-sample={}-k={}-epochs={}-' \
                           'lr={}-wd={}-exp={}'.format(args.test_target, args.classifier, args.dataset, args.model,
                                                args.mislabel_rate, args.noise_type, args.sample_rate, args.k,
                                                args.n_epochs, args.lr, args.weight_decay, args.exp)

    # Step 1: train GNN and record the sm / log_sm vectors
    old_all_sm_vectors, all_two_logits, best_sm_vectors, data, n_classes = train_GNNs(args.model, args.dataset, args.n_epochs, args.lr,
                                                                     args.weight_decay, trained_model_file,
                                                                     args.mislabel_rate, args.noise_type, args.batch_size)
    print("Held split is ", args.held_split)
    data.held_mask = data.train_mask if args.held_split == 'train' else data.val_mask if args.held_split == 'valid' else data.test_mask

    # Step 2: do negative sampling randomly
    noisy_data, tr_sample_idx = negative_sampling(data, args.sample_rate, n_classes)
    sample_ratio = len(tr_sample_idx) / sum(data.held_mask)
    print("{} negative samples in {} set, with a sample ratio of {}".format(
        len(tr_sample_idx), sum(data.held_mask), sample_ratio))

    features, y = generate_new_feature(args.k, data, noisy_data, tr_sample_idx, old_all_sm_vectors, all_two_logits, n_classes)
    X_train = features[data.held_mask].reshape(features[data.held_mask].shape[0], -1)
    y_train = y[data.held_mask]
    if args.test_target == 'valid':
        X_test = features[data.val_mask].reshape(features[data.val_mask].shape[0], -1)
        print("Test target is {} with {} data.".format(args.test_target, sum(data.val_mask)))
    elif args.test_target == 'test':
        X_test = features[data.test_mask].reshape(features[data.test_mask].shape[0], -1)
        print("Test target is {} with {} data.".format(args.test_target, sum(data.test_mask)))
    if args.classifier == "LR":
        classifier = LogisticRegression(max_iter=3000, multi_class='ovr', verbose=True)
    elif args.classifier == "XGB":
        classifier = XGBClassifier()
    elif args.classifier == "RF":
        classifier = RandomForestClassifier()
    elif args.classifier == "SVC":
        classifier = SVC(probability=True)
    elif args.classifier == "KNN":
        classifier = KNeighborsClassifier()
    elif args.classifier == "MLP":
        classifier = myMLP([X_train.shape[1], 32, 2])
    print("Fitting with {}......".format(args.classifier))
    if args.classifier != "MLP":
        classifier.fit(X_train, y_train)
    else:
        ensure_dir('tensorboard_logs')
        log_dir = 'tensorboard_logs/MLP-noCL-{}-{}-mislabel={}-{}-sample={}-k={}-epochs={}-lr={}-wd={}-exp={}'.format \
                    (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.sample_rate, args.k,
                    500, 0.001, args.weight_decay, args.exp)
        writer = create_summary_writer(log_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ", device)
        X_train = torch.from_numpy(X_train).float().to(device)
        y_train = torch.from_numpy(y_train).long().to(device)
        classifier.to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=args.weight_decay)
        for epoch in range(500):
            classifier.train()
            optimizer.zero_grad()
            out = classifier(X_train)
            loss = torch.nn.L1Loss()(out[:, 1], y_train.float())
            acc = accuracy_score(y_train.cpu().detach().numpy(), out.cpu().detach().numpy()[:, 1] > .5)
            print("Epoch[{}] Tr Loss: {:.2f} Acc: {:.2f}".format(epoch + 1, loss, acc))
            writer.add_scalar("MLP_training/loss", loss, epoch)
            loss.backward()
            optimizer.step()

    # Step 4: predict on the validation set
    print("Predicting......")
    if args.classifier != "MLP":
        probs = classifier.predict_proba(X_test)[:, 1]
    else:
        classifier.eval()
        probs = classifier(X_train).cpu().detach().numpy()[:, 1]
        roc_auc = roc_auc_score(y_train.cpu().detach().numpy(), probs)
        print("ROC AUC Score on Training set: {:.2f}".format(roc_auc))
        X_test = torch.from_numpy(X_test).float().to(device)
        probs = classifier(X_test).cpu().detach().numpy()[:, 1]
    print("Saving result......")
    idx2prob = dict(zip([i for i in range(len(probs))], probs))
    result = probs > 0.97
    idx2score = dict()
    for i in range(len(result)):
        if result[i]:
            idx2score[i] = probs[i]
    er = [x[0] for x in sorted(idx2score.items(), key=lambda x: x[1], reverse=True)]
    cl_results = pd.DataFrame({'result': pd.Series(result), 'ordered_errors': pd.Series(er), 'score': pd.Series(probs)})
    cl_results.to_csv(mislabel_result_file+'.csv', index=False)
    ordered_idx = [x[0] for x in sorted(idx2prob.items(), key=lambda x: x[1], reverse=True)]
    ytest = get_ytest(args.dataset, args.noise_type, args.mislabel_rate, args.test_target)
    cal_patk(ordered_idx, ytest)
    cal_mcc(result, ytest)
    cal_afpr(result, ytest)
    roc_auc = roc_auc_score(ytest, probs)
    print("ROC AUC Score on Test set: {:.2f}".format(roc_auc))
