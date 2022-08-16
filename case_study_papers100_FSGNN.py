import sys
import copy
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F

from cleanlab.latent_estimation import compute_confident_joint, estimate_latent
from Utils import setup_seed
from run_CL_negsamp import negative_sampling, generate_new_feature_for_case_study, myMLP
from case_study_baseclassifier import sampleSubGraph
from torch_geometric.utils import degree, to_undirected


if __name__ == '__main__':
    setup_seed(1119)
    parser = argparse.ArgumentParser(description="Our Approach")
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument("--sample_rate", type=float, default=0.5)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    origin = sys.stdout
    f = open('papers100_case_study_FSGNN_train.txt', 'w')
    sys.stdout = f

    # load data
    dataname = 'ogbn-papers100M'
    data, n_nodes, n_classes = sampleSubGraph()
    data.y = data.y.squeeze()

    # the split idx is global idx
    train_idx = torch.nonzero(data.train_mask == True)[:, 0]
    val_idx = torch.nonzero(data.val_mask == True)[:, 0]
    test_idx = torch.nonzero(data.test_mask == True)[:, 0]
    data.held_mask = data.val_mask
    data.used_mask = torch.ones(n_nodes, dtype=torch.bool)

    # Step 1: load pred
    ori_pred = np.load('train_output_seed1.npy')
    print("ori_pred shape: ", ori_pred.shape)

    mapping = pd.read_csv("./dataset/ogbn_{}/mapping/nodeidx2paperid.csv".format(dataname[5:]), index_col=0)
    catmap = pd.read_csv("./dataset/ogbn_{}/mapping/labelidx2arxivcategeory.csv".format(dataname[5:]), index_col=0)
    with open('./output/oldy2newy_train', 'rb') as file:
        oldy2newy = pickle.load(file)
    with open('./output/idx2ogbidx_train', 'rb') as file:
        idx2ogbidx = pickle.load(file)
    with open('./output/idx2testidx_train', 'rb') as file:
        idx2testidx = pickle.load(file)

    pred = np.zeros((ori_pred.shape[0], len(oldy2newy)))
    for oldy, newy in oldy2newy.items():
        pred[:, newy] = ori_pred[:, oldy]
    pred = pred[list(idx2testidx.values())]
    pred = F.softmax(torch.tensor(pred), dim=1).numpy()
    print("pred shape: ", pred.shape)
    all_sm_vectors = copy.deepcopy(pred)
    all_sm_vectors = all_sm_vectors[np.newaxis]

    # Step 2: calculate confident joint and generate noise
    confident_joint = compute_confident_joint(data.y[data.val_mask], pred[data.val_mask])
    print("Confident Joint: ", confident_joint)
    py, noise_matrix, inv_noise_matrix = estimate_latent(confident_joint, data.y[data.val_mask])
    print("Noise Matrix (p(s|y)): ")
    print(noise_matrix)
    plt.figure()
    sns.heatmap(noise_matrix.T, cmap='PuBu', vmin=0, vmax=1, linewidth=1, annot=n_classes < 10)
    plt.title('Learned Noise Transition Matrix')
    plt.savefig('Case_Study_Noise_Matrix_FSGNN_train_' + dataname + '.jpg', bbox_inches='tight')
    plt.show()

    noisy_data, tr_sample_idx = negative_sampling(data, noise_matrix, args.sample_rate, n_classes)
    sample_ratio = len(tr_sample_idx) / sum(data.held_mask)
    print("{} negative samples in {} set, with a sample ratio of {}".format(
        len(tr_sample_idx), sum(data.held_mask), sample_ratio))

    # Step 3: fit a classifier with the combined feature
    features, y = generate_new_feature_for_case_study(args.k, data, noisy_data, tr_sample_idx, all_sm_vectors, None, n_classes)
    X_train = features[data.held_mask].reshape(features[data.held_mask].shape[0], -1)
    y_train = y[data.held_mask]

    classifier = myMLP([X_train.shape[1], 32, 2])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=0.0005)
    for epoch in range(args.n_epochs):
        classifier.train()
        optimizer.zero_grad()
        out = classifier(X_train)
        loss = torch.nn.L1Loss()(out[:, 1], y_train.float())
        acc = accuracy_score(y_train.cpu().detach().numpy(), out.cpu().detach().numpy()[:, 1] > .5)
        print("Epoch[{}] Tr Loss: {:.2f} Acc: {:.2f}".format(epoch + 1, loss, acc))
        loss.backward()
        optimizer.step()

    # Step 4: predict on the split test set
    classifier.eval()
    probs = classifier(X_train).cpu().detach().numpy()[:, 1]
    roc_auc = roc_auc_score(y_train.cpu().detach().numpy(), probs)
    print("ROC AUC Score on Training set: {:.2f}".format(roc_auc))

    X_test = features[test_idx].reshape(features[test_idx].shape[0], -1)
    X_test = torch.from_numpy(X_test).float().to(device)
    probs = classifier(X_test).cpu().detach().numpy()[:, 1]

    # Step 5: analysis
    pred = pred[test_idx]
    predmax = np.argmax(pred, axis=1)
    # labels = np.squeeze(data.y[test_idx].numpy())
    labels = [int(i) for i in data.y[test_idx]]

    cert = pred[np.arange(len(labels)), predmax]
    cert_correct = pred[np.arange(len(labels)), labels]
    sort_idx = np.argsort(-probs)
    newy2oldy = dict(zip(oldy2newy.values(), oldy2newy.keys()))
    print("oldy2newy: ", oldy2newy)
    print("newy2oldy: ", newy2oldy)

    edge_index = to_undirected(data.edge_index)
    degree = degree(edge_index[0])
    for i in range(200):
        t_idx = sort_idx[i]  # test set indices
        tes_idx = int(test_idx[t_idx])
        if degree[tes_idx] >= 3 :
            ogb_idx = idx2ogbidx[tes_idx]  # indices in ogb dataset change!!!
            cur_pred = predmax[t_idx]
            cur_label = labels[t_idx]
            print("{} (test {} ogb {} arxiv {}): pred = {}[{}] ({:.5f}) actual = {}[{}] ({:.5f}) mislabel prob = {}".format(
                i, t_idx, ogb_idx, mapping.loc[ogb_idx, 'paper id'], newy2oldy[cur_pred],
                catmap.loc[newy2oldy[cur_pred], 'arxiv category'], cert[t_idx], newy2oldy[cur_label],
                catmap.loc[newy2oldy[cur_label], 'arxiv category'], cert_correct[t_idx], probs[t_idx]))

    sys.stdout = origin
    f.close()
