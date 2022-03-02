import os
import copy
from tqdm import tqdm
import random
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, roc_curve

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader, GraphSAINTRandomWalkSampler

from GNN_models import GCN, myGIN, GAT, baseMLP
from Utils import to_softmax, get_data, create_summary_writer, setup_seed, ensure_dir
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
    scheduler = ReduceLROnPlateau(optimizer, mode='max')
    if dataset in ['Reddit2']:  # cluster data for big graph
        # cluster_data = ClusterData(data, num_parts=1024, recursive=False)
        # train_loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True)
        train_loader = GraphSAINTRandomWalkSampler(data, batch_size=6000, walk_length=2, num_steps=5,
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
        if dataset in ['Flickr', 'Cora', 'CiteSeer', 'PubMed']:  # small graph
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

            # just try using the logit and the largest other logit
            all_logits = model.get_logits(data).cpu().detach().numpy()
            normalized_all_logits = (all_logits - all_logits.min(axis=0)) / (all_logits.max(axis=0) - all_logits.min(axis=0))
            all_y = data.y.cpu().detach().numpy()
            aum = []
            for i, y in enumerate(all_y):
                tmp = normalized_all_logits[i]
                logit = tmp[y]
                np.delete(tmp, [y])
                aum.append([logit, max(tmp)])
            all_two_logits.append(aum)
        # scheduler.step(cri)

    return np.array(all_sm_vectors), np.array(all_two_logits), to_softmax(best_sm_vectors), data.cpu().detach(), n_classes


def generate_khop_neighbor_feature(G, node, k, label_vector):
    feature = []
    nb2length = nx.single_source_shortest_path_length(G, node, cutoff=k)
    length2nb = dict(zip([i for i in range(k+1)], [[] for _ in range(k+1)]))
    for nb, l in nb2length.items():
        length2nb[l].append(nb)
    for hop in range(k+1):
        if len(length2nb[hop]) != 0:
            feature.append(np.sum(label_vector[length2nb[hop]], axis=0) / len(length2nb[hop]))
            # feature.append(np.sum(label_vector[length2nb[hop]], axis=0))  # unnormalize to make agg2 correct
        else:
            print("Warning: Node {} does not have {} hop neighbors!".format(node, hop))
            feature.append(np.zeros(len(label_vector[0])))
    return np.array(feature)


def generate_sm_feature(idx, all_sm_vectors):
    return np.array(all_sm_vectors[:, idx])


def generate_new_feature(k, data, all_sm_vectors, all_two_logits, n_classes):
    print("Generating new features......")
    # construct an undirected graph
    G = nx.Graph()
    for i in range(len(data.edge_index[0])):
        G.add_edge(data.edge_index[0][i].item(), data.edge_index[1][i].item())

    # get the one-hot label vector
    label_vector = np.array([np.zeros(n_classes)] * data.num_nodes)
    for i, y in enumerate(data.y):
        label_vector[i][y] = 1

    # combine the k-hop neighbor feature and the sm feature
    features = [[]] * data.num_nodes
    for node in G.nodes():
        khop_feature = generate_khop_neighbor_feature(G, node, k, label_vector)
        sm_feature = generate_sm_feature(node, all_sm_vectors)

        agg0 = np.sum(to_softmax(sm_feature)[-1,:] * khop_feature[0])  # softmax score for observed label
        # agg0 = to_softmax(sm_feature)[-1,:] * khop_feature[0].reshape(-1)
        agg1 = np.sum(khop_feature[1] * khop_feature[0])  # neighbour count for observed label
        # agg2 = np.sum((khop_feature[1] / np.sum(khop_feature[1])) * khop_feature[0])  # fraction of neighbors with observed label
        agg2 = np.average(all_two_logits[:,node,0] - all_two_logits[:,node,1])
        features[node] = np.hstack((agg0, agg1))

    # deal with orphans
    orphan_cnt = 0
    for i,feat in enumerate(features):
        if len(feat) == 0:
            orphan_cnt += 1
            print("Deal with orphan")
            features[i] = np.zeros(2)
            sm_feature = generate_sm_feature(i, all_sm_vectors)
            khop_feature = label_vector[i]
            features[i][0] = np.sum(to_softmax(sm_feature)[-1,:] * khop_feature)
            print("orphan feature: ", features[i])
    print("{} orphans in total!".format(orphan_cnt))

    return np.array(features)


if __name__ == "__main__":
    setup_seed(1119)

    parser = argparse.ArgumentParser(description="Our Approach")
    parser.add_argument("--dataset", type=str, default='Cora')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--mislabel_rate", type=float, default=0.1)
    parser.add_argument("--noise_type", type=str, default='symmetric')
    parser.add_argument("--sample_rate", type=float, default=0.5)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--classifier", type=str, default='MLP')
    parser.add_argument("--test_target", type=str, default='test')
    parser.add_argument("--lamda", type=float, default=1)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--validation', type=bool, default=True)
    args = parser.parse_args()

    ensure_dir('checkpoints')
    trained_model_file = 'checkpoints/{}-{}-mislabel={}-{}-epochs={}-bs={}-lr={}-wd={}'.format\
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.batch_size, args.lr,
         args.weight_decay)
    ensure_dir('mislabel_results')
    mislabel_result_file = 'mislabel_results/shrunk-lamda={}-test={}-{}-{}-{}-mislabel={}-{}-sample={}-k={}-epochs={}-' \
                           'lr={}-wd={}'.format(args.lamda, args.test_target, args.classifier, args.dataset, args.model,
                                                args.mislabel_rate, args.noise_type, args.sample_rate, args.k,
                                                args.n_epochs, args.lr, args.weight_decay)

    # Step 1: train GNN and record the sm / log_sm vectors
    old_all_sm_vectors, all_two_logits, best_sm_vectors, data, n_classes = train_GNNs(args.model, args.dataset, args.n_epochs, args.lr,
                                                                     args.weight_decay, trained_model_file,
                                                                     args.mislabel_rate, args.noise_type, args.batch_size)

    # Step 2: fit a classifier with the combined feature
    features = generate_new_feature(args.k, data, old_all_sm_vectors, all_two_logits, n_classes)

    # Step 3: predict on the test set
    print("Predicting......")
    agg0 = features[data.test_mask][:,0]
    agg1 = features[data.test_mask][:,1]
    score = -(agg0 + args.lamda * agg1)
    score = (score - min(score)) / (max(score) - min(score))
    print("Saving result......")
    idx2prob = dict(zip([i for i in range(len(score))], score))
    result = score > 0.5
    idx2score = dict()
    for i in range(len(result)):
        if result[i]:
            idx2score[i] = score[i]
    er = [x[0] for x in sorted(idx2score.items(), key=lambda x: x[1], reverse=True)]
    cl_results = pd.DataFrame({'result': pd.Series(result), 'ordered_errors': pd.Series(er), 'score': pd.Series(score)})
    cl_results.to_csv(mislabel_result_file+'.csv', index=False)
    ordered_idx = [x[0] for x in sorted(idx2prob.items(), key=lambda x: x[1], reverse=True)]
    ytest = get_ytest(args.dataset, args.noise_type, args.mislabel_rate, args.test_target)
    cal_patk(ordered_idx, ytest)
    cal_mcc(result, ytest)
    cal_afpr(result, ytest)
    roc_auc = roc_auc_score(ytest, score)
    fpr, tpr, thresholds = roc_curve(ytest, score)
    # print("fpr: ", fpr)
    # print("tpr: ", tpr)
    # print("thresholds: ", thresholds)
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.show()
    print("ROC AUC Score on Test set: {:.2f}".format(roc_auc))
