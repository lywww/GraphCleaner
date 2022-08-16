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
import networkx as nx

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

import torch
import torch_geometric
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader, GraphSAINTRandomWalkSampler
from torch_geometric.nn.models import MLP

from GNN_models import GCN, myGIN, GAT, baseMLP
from Utils import to_softmax, get_data, create_summary_writer, ensure_dir, setup_seed
from cleanlab.latent_estimation import compute_confident_joint, estimate_latent
from evaluate_different_methods import cal_patk, cal_afpr, get_ytest, cal_mcc


def train_GNNs(model_name, dataset, n_epochs, lr, wd, trained_model_file, mislabel_rate, noise_type, batch_size, neg_data=None):
    # prepare data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    data, n_classes = get_data(dataset, noise_type, mislabel_rate)
    if neg_data:
        data = neg_data
    # data.to(device)
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
    # scheduler = ReduceLROnPlateau(optimizer, mode='max')
    if dataset in ['Reddit2', 'ogbn-papers100M']:  # cluster data for big graph
        # cluster_data = ClusterData(data, num_parts=1024, recursive=False)
        # train_loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True)
        train_loader = GraphSAINTRandomWalkSampler(data, batch_size=batch_size, walk_length=2, num_steps=5,
                                                   sample_coverage=100, shuffle=True)
        data_loader = NeighborLoader(copy.copy(data), input_nodes=None, num_neighbors=[-1], batch_size=batch_size,
                                     shuffle=False, num_workers=1)
        # No need to maintain these features during evaluation:
        del data_loader.data.x, data_loader.data.y
        # Add global node index information.
        data_loader.data.num_nodes = data.num_nodes
        data_loader.data.n_id = torch.arange(data.num_nodes)

    # load / train the model
    all_sm_vectors = []
    all_two_logits = []
    best_sm_vectors = []
    best_cri = 0
    if neg_data:
        model.load_state_dict(torch.load(trained_model_file))
    for epoch in range(n_epochs):
        model.train()
        if dataset in ['Flickr', 'Cora', 'CiteSeer', 'PubMed', 'ogbn-arxiv', 'Computers', 'Photo']:  # small graph
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

        elif dataset in ['Reddit2']:  # large graph
            total_loss = total_nodes = 0
            for batch in train_loader:
                # print(batch)
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch.num_nodes
                total_nodes += batch.num_nodes
            print("Epoch[{}] Loss: {:.2f}".format(epoch + 1, total_loss / total_nodes))

            if (epoch+1) % 20 == 0:
                model.eval()
                eval_out = []
                pbar = tqdm(total=len(data_loader.dataset))
                pbar.set_description('Evaluating')
                for batch in data_loader:
                    x = data.x[batch.n_id].to(device)
                    out = model(Data(x, batch.edge_index.to(device)))
                    eval_out.append(out[:batch.batch_size].cpu().detach())
                    torch.cuda.empty_cache()
                    pbar.update(batch.batch_size)
                pbar.close()
                eval_out = torch.cat(eval_out, dim=0)
                y_pred = eval_out[data.val_mask].argmax(dim=-1)
                y_true = data.y[data.val_mask]
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


def generate_khop_neighbor_feature(G, node, k, label_vector):
    feature = []
    nb2length = nx.single_source_shortest_path_length(G, node, cutoff=k)
    length2nb = dict(zip([i for i in range(k+1)], [[] for _ in range(k+1)]))
    for nb, l in nb2length.items():
        length2nb[l].append(nb)
    for hop in range(k+1):
        if len(length2nb[hop]) != 0:
            feature.append(np.sum(label_vector[length2nb[hop]], axis=0) / len(length2nb[hop]))
        else:
            print("Warning: Node {} does not have {} hop neighbors!".format(node, hop))
            feature.append(np.zeros(len(label_vector[0])))
            # feature.append(np.ones(len(label_vector[0])) / len(label_vector[0]))
    return np.array(feature)


def generate_sm_feature(idx, all_sm_vectors):
    return np.array(all_sm_vectors[:, idx])


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
    # print("ymat: ", ymat)
    L0 = np.eye(n_classes)[data.y]  # original labels, converted to one-hot matrix
    L_corr = (ymat * np.eye(n_classes)[noisy_data.y]) + (1 - ymat) * L0  # label matrix corrupted by negative samples
    # print("L_corr:", L_corr)
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


# def generate_new_feature(k, data, noisy_data, sample_idx, all_sm_vectors, all_two_logits, n_classes):
#     print("Generating new features......")
#     # construct an undirected graph
#     G = nx.Graph()
#     for i in range(len(data.edge_index[0])):
#         G.add_edge(data.edge_index[0][i].item(), data.edge_index[1][i].item())
#
#     # get the one-hot label vector
#     label_vector = np.array([np.zeros(n_classes)] * data.num_nodes)
#     for i, y in enumerate(data.y):
#         label_vector[i][y] = 1
#
#     # combine the k-hop neighbor feature and the sm feature
#     features = [[]] * data.num_nodes
#     y = np.zeros(data.num_nodes)  # 1 indicates negative / noisy
#     y[sample_idx] = 1
#     ident = np.eye(n_classes)
#     for node in G.nodes():
#         khop_feature = generate_khop_neighbor_feature(G, node, k, label_vector)
#         if y[node] == 1:
#             # negative samples only modify their own feature, not features of other nodes
#             # here we modify its feature to the noisy version
#             khop_feature[0] = np.zeros_like(khop_feature[0]) + ident[noisy_data.y[node]]
#         sm_feature = generate_sm_feature(node, all_sm_vectors)
#         # features[node] = np.vstack((khop_feature, sm_feature))  # (k + No. of selected epochs, n_classes)
#         # features[node] = np.hstack((khop_feature.reshape(-1), sm_feature[-1,:].reshape(-1)))
#
#         agg0 = np.sum(to_softmax(sm_feature)[-20:0,:] * khop_feature[0], axis=1)  # softmax score for observed label
#         # agg0 = to_softmax(sm_feature)[-1,:] * khop_feature[0].reshape(-1)
#         agg1 = np.sum(khop_feature[1:,:] * khop_feature[0], axis=1)  # neighbour count for observed label
#         # agg2 = np.sum((khop_feature[1] / np.sum(khop_feature[1])) * khop_feature[0])  # fraction of neighbors with observed label
#         agg2 = np.average(all_two_logits[:,node,0] - all_two_logits[:,node,1])
#         features[node] = np.hstack((agg0, agg1))
#
#         # features[node] = np.hstack((agg0, agg1, agg2))
#
#         # just try using the logit and the largest other logit
#         # features[node.item()] = np.hstack((khop_feature.reshape(-1), sm_feature))
#
#     # deal with orphans
#     orphan_cnt = 0
#     for i,feat in enumerate(features):
#         if len(feat) == 0:
#             orphan_cnt += 1
#             print("Deal with orphan")
#             features[i] = np.zeros(2)
#             sm_feature = generate_sm_feature(i, all_sm_vectors)
#             khop_feature = label_vector[i]
#             if y[i] == 1:
#                 khop_feature = np.zeros_like(khop_feature) + ident[noisy_data.y[i]]
#             features[i][0] = np.sum(to_softmax(sm_feature)[-1,:] * khop_feature)
#             print("orphan feature: ", features[i])
#     print("{} orphans in total!".format(orphan_cnt))
#
#     return np.array(features), y


class myMLP(torch.nn.Module):
    def __init__(self, channel_list, dropout=0.0, relu_first=True):
        super().__init__()
        self.mlp = MLP(channel_list=channel_list, dropout=dropout, relu_first=relu_first, batch_norm=True)

    def forward(self, data):
        f = self.mlp(data)
        y = F.softmax(f, dim=1)
        return y


if __name__ == "__main__":
    setup_seed(1119)

    parser = argparse.ArgumentParser(description="Our Approach")
    parser.add_argument("--exp", type=int, default=0)
    parser.add_argument("--dataset", type=str, default='Cora')  # Reddit2 not usable currently
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--mislabel_rate", type=float, default=0.1)
    parser.add_argument("--noise_type", type=str, default='symmetric')
    parser.add_argument("--sample_rate", type=float, default=0.5)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--classifier", type=str, default='MLP')
    parser.add_argument("--held_split", type=str, default='valid')
    parser.add_argument("--test_target", type=str, default='test')
    parser.add_argument("--n_epochs", type=int, default=200, help='Planetoid:200; ogbn-arxiv:500')
    parser.add_argument("--lr", type=float, default=0.001, help='Planetoid:0.001; ogbn-arxiv:0.001')
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

    # Step 3: fit a classifier with the combined feature
    # all_sm_vectors, best_sm_vectors, data, n_classes = train_GNNs(args.model, args.dataset, 200, args.lr,
    #                                                               args.weight_decay, trained_model_file,
    #                                                               args.mislabel_rate, args.noise_type, args.batch_size,
    #                                                               noisy_data)
    # all_sm_vectors = np.vstack((old_all_sm_vectors, new_all_sm_vectors))
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
        classifier = myMLP([X_train.shape[1], 32, 2])  # 32
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
            # loss = F.nll_loss(torch.log(out), y_train)
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

    # fpr, tpr, thresholds = roc_curve(ytest, probs)
    # plt.plot(fpr, tpr)
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    # plt.title('ROC of ' + args.dataset)
    # plt.savefig('ROC' + '_' + args.dataset + '_' + args.noise_type + '_' + str(args.mislabel_rate) + '_' + args.model +
    #             '_' + str(args.k) + '.jpg', bbox_inches='tight')
    # plt.show()
