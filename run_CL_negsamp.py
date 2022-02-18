import os
import copy
from tqdm import tqdm
import random
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sklearn
import scipy

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from scipy.sparse import spdiags

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader, GraphSAINTRandomWalkSampler
from torch_geometric.nn.models import MLP

from run_GNNs_fordrawing import GCN, myGIN, GAT
from run_GNNs import to_softmax, get_data, create_summary_writer
from cleanlab.latent_estimation import compute_confident_joint, estimate_latent
from evaluate_different_methods import cal_patk, cal_afpr, get_ytest


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train_GNNs(model_name, dataset, n_epochs, lr, wd, trained_model_file, mislabel_rate, noise_type, batch_size, neg_data=None):
    # prepare data
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
                    # print("out:", out)
                    # print("out size:", out.size())
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
        # all_sm_vectors.append(F.nll_loss(out, data.y, reduction='none').cpu().detach().numpy())

        # just try using the logit and the largest other logit
        # all_logits = model.get_logits(data).cpu().detach().numpy()
        # normalized_all_logits = (all_logits - all_logits.min(axis=0)) / (all_logits.max(axis=0) - all_logits.min(axis=0))
        # all_y = data.y.cpu().detach().numpy()
        # aum = []
        # for i, y in enumerate(all_y):
        #     tmp = normalized_all_logits[i]
        #     logit = tmp[y]
        #     np.delete(tmp, [y])
        #     aum.append(logit-max(tmp))
        # all_sm_vectors.append(aum)
        # scheduler.step(cri)

    return np.array(all_sm_vectors), to_softmax(best_sm_vectors), data.cpu().detach(), n_classes


def negative_sampling(data_orig, noise_matrix, sample_rate, n_classes):
    # filter out the classes with invalid noise transition probability distribution
    data = copy.deepcopy(data_orig)
    train_idx = np.argwhere(data.held_mask == True)[0]
    train_y = data.y[data.held_mask]
    valid_subidx = set([i for i in range(len(train_y))])
    for c in range(n_classes):
        if np.isnan(noise_matrix[0][c]) or max(noise_matrix[:, c]) == 1:
            print("Class {} is invalid!".format(c))
            valid = set(np.argwhere(train_y != c)[0].numpy())
            valid_subidx = valid_subidx & valid
    train_idx = train_idx[list(valid_subidx)]

    # sampling and change classes
    tr_sample_idx = random.sample(list(train_idx), int(np.round(sample_rate * len(train_idx))))
    for idx in tr_sample_idx:
        y = data.y[idx]
        while y == data.y[idx]:
            y = np.random.choice([i for i in range(n_classes)], p=noise_matrix[:,y])
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
            feature.append(np.sum(label_vector[length2nb[hop]], axis=0))
        else:
            print("Warning: Node {} does not have {} hop neighbors!".format(node, hop))
            feature.append(np.zeros(len(label_vector[0])))
    return np.array(feature)


def generate_sm_feature(idx, all_sm_vectors):
    return np.array(all_sm_vectors[:, idx])


def generate_new_feature(k, data, noisy_data, sample_idx, all_sm_vectors, n_classes):
    print("Generating new features......")

    y = np.zeros(data.num_nodes)  # 1 indicates negative / noisy
    y[sample_idx] = 1

    A = torch_geometric.utils.convert.to_scipy_sparse_matrix(data.edge_index)
    n = A.shape[0]

    # symmetric normalized adjacency matrix. 1e-10 is to prevent dividing by zero.
    S = spdiags(np.squeeze((1e-10 + np.array(A.sum(1)))**-0.5), 0, n, n) @ A @ spdiags(np.squeeze((1e-10 + np.array(A.sum(0)))**-0.5), 0, n, n)
    S2 = S @ S
    S3 = S @ S2
    S2.setdiag(np.zeros((n,))) # remove self-loops to prevent the algorithm peeking at the original label (L0)
    S3.setdiag(np.zeros((n,)))

    ymat = y[:, np.newaxis]
    L0 = np.eye(n_classes)[data.y] # original labels, converted to one-hot matrix
    L_corr = (ymat * np.eye(n_classes)[noisy_data.y]) + (1 - ymat) * L0 # label matrix corrupted by negative samples
    L1 = S @ L0
    L2 = S2 @ L0
    L3 = S3 @ L0

    P0 = scipy.special.softmax(all_sm_vectors[-1, :, :], axis=1) # base model softmax predictions matrix
    P1 = S @ P0
    P2 = S2 @ P0
    P3 = S3 @ P0

    feat = np.hstack((
        np.sum(L_corr * P0, axis=1, keepdims=True), # since L_corr is one-hot, this just extracts the corresponding entry of P0
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
    setup_seed(1119)

    parser = argparse.ArgumentParser(description="Our Approach")
    parser.add_argument("--dataset", type=str, default='PubMed')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--mislabel_rate", type=float, default=0.1)
    parser.add_argument("--noise_type", type=str, default='symmetric')
    parser.add_argument("--sample_rate", type=float, default=0.5)
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--classifier", type=str, default='MLP')
    parser.add_argument("--held_split", type=str, default='valid') # train/valid/test
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--validation', type=bool, default=False)
    args = parser.parse_args()

    ensure_dir('checkpoints')
    trained_model_file = 'checkpoints/{}-{}-mislabel={}-{}-epochs={}-bs={}-lr={}-wd={}'.format\
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.batch_size, args.lr,
         args.weight_decay)
    ensure_dir('mislabel_results')
    mislabel_result_file = 'mislabel_results/valtr-newloss-{}-{}-{}-mislabel={}-{}-sample={}-k={}-epochs={}-lr={}-wd={}'.format \
        (args.classifier, args.dataset, args.model, args.mislabel_rate, args.noise_type, args.sample_rate, args.k,
         args.n_epochs, args.lr, args.weight_decay)

    # Step 1: train GNN and record the sm / log_sm vectors
    old_all_sm_vectors, best_sm_vectors, data, n_classes = train_GNNs(args.model, args.dataset, args.n_epochs, args.lr,
                                                                     args.weight_decay, trained_model_file,
                                                                     args.mislabel_rate, args.noise_type, args.batch_size)

    print("Held split is ", args.held_split)
    data.held_mask = data.train_mask if args.held_split == 'train' else data.val_mask if args.held_split == 'valid' else data.test_mask
    # Step 2: calculate confident joint and generate noise
    confident_joint = compute_confident_joint(data.y[data.val_mask], best_sm_vectors[data.val_mask])
    print("Confident Joint: ", confident_joint)
    py, noise_matrix, inv_noise_matrix = estimate_latent(confident_joint, data.y[data.val_mask])
    print("Noise Matrix (p(s|y)): ")
    print(noise_matrix)
    # plt.figure()
    # sns.heatmap(noise_matrix.T, cmap='PuBu', vmin=0, vmax=1, linewidth=1, annot=True)
    # plt.title('Learned Noise Transition Matrix')
    # plt.savefig('val_Noise_Matrix_'+args.dataset+'_'+args.noise_type+'_'+str(args.mislabel_rate)+'_'+args.model+'.jpg',
    #             bbox_inches='tight')
    # plt.show()

    ##
    # matrix_name = 'GT_Noise_Matrix_' + args.dataset + '_' + args.noise_type + '_' + str(args.mislabel_rate)
    # noise_matrix = np.load(matrix_name + '.npy')

    # sns.heatmap(noise_matrix.T, cmap='PuBu', vmin=0, vmax=1, linewidth=1, annot=True)
    # plt.title('Oracle Noise Transition Matrix')
    # plt.show()
    # print("Oracle")
    # print(noise_matrix)
    noisy_data, tr_sample_idx = negative_sampling(data, noise_matrix, args.sample_rate, n_classes)
    sample_ratio = len(tr_sample_idx) / sum(data.held_mask)
    print("{} negative samples in {} set, with a sample ratio of {}".format(
        len(tr_sample_idx), sum(data.held_mask), sample_ratio))

    # Step 3: fit a classifier with the combined feature
    # all_sm_vectors, best_sm_vectors, data, n_classes = train_GNNs(args.model, args.dataset, 200, args.lr,
    #                                                               args.weight_decay, trained_model_file,
    #                                                               args.mislabel_rate, args.noise_type, args.batch_size,
    #                                                               noisy_data)
    # print("old sm: ", old_all_sm_vectors)
    # print("new sm: ", new_all_sm_vectors)
    # all_sm_vectors = np.vstack((old_all_sm_vectors, new_all_sm_vectors))

    # note: generated features are clean for all points but corrupted for the negative samples in held_mask (e.g. val set)
    features, y = generate_new_feature(args.k, data, noisy_data, tr_sample_idx, old_all_sm_vectors, n_classes)
    # features = sklearn.preprocessing.StandardScaler().fit_transform(features)

    ##
    X_train = features[data.held_mask].reshape(features[data.held_mask].shape[0], -1)
    y_train = y[data.held_mask]
    X_test = features[data.test_mask].reshape(features[data.test_mask].shape[0], -1)
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
        log_dir = 'tensorboard_logs/MLP-{}-{}-mislabel={}-{}-sample={}-k={}-epochs={}-lr={}-wd={}'.format \
                    (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.sample_rate, args.k,
                    600, 0.001, args.weight_decay)
        writer = create_summary_writer(log_dir)
        X_train = torch.from_numpy(X_train).float().to(device)
        # y_train = np.stack([1-y_train, y_train], axis=1)
        y_train = torch.from_numpy(y_train).long().to(device)
        classifier.to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=args.weight_decay)
        for epoch in range(500):  #600
            classifier.train()
            optimizer.zero_grad()
            out = classifier(X_train)
            loss = torch.nn.L1Loss()(out[:,1], y_train.float())
            acc = accuracy_score(y_train.cpu().detach().numpy(), out.cpu().detach().numpy()[:, 1] > .5)
            # te_acc = accuracy_score(y_train, out.cpu().detach().numpy()[:, 1] > .5)
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
    ##

    print("Saving result......")
    idx2prob = dict(zip([i for i in range(len(probs))], probs))
    result = probs > 0.5
    idx2score = dict()
    for i in range(len(result)):
        if result[i]:
            idx2score[i] = probs[i]
    er = [x[0] for x in sorted(idx2score.items(), key=lambda x: x[1], reverse=True)]
    cl_results = pd.DataFrame({'result': pd.Series(result), 'ordered_errors': pd.Series(er), 'score': pd.Series(probs)})
    # cl_results = pd.DataFrame({'Id': [i for i in range(len(probs))], 'Prob': probs})
    cl_results.to_csv(mislabel_result_file+'.csv', index=False)
    ordered_idx = [x[0] for x in sorted(idx2prob.items(), key=lambda x: x[1], reverse=True)]
    ytest = get_ytest(args.dataset, args.noise_type, args.mislabel_rate, args.validation)
    cal_patk(ordered_idx, ytest)
    cal_afpr(result, ytest)
    roc_auc = roc_auc_score(ytest, probs)
    print("ROC AUC Score on Test set: {:.2f}".format(roc_auc))
    print("on dataset: ", args.dataset)