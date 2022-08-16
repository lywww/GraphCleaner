import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F

# from logger import Logger
from ogb.nodeproppred import PygNodePropPredDataset
from cleanlab.latent_estimation import compute_confident_joint, estimate_latent

from Utils import setup_seed
from run_CL_negsamp import negative_sampling, generate_new_feature_for_case_study, myMLP


if __name__ == '__main__':
    setup_seed(1119)
    parser = argparse.ArgumentParser(description="Our Approach")
    parser.add_argument("--dataset", type=str, default='ogbn-arxiv')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument("--sample_rate", type=float, default=0.5)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    dataname = args.dataset
    print("dataname:", dataname)
    dataset = PygNodePropPredDataset(name=dataname, root='./dataset/')
    ori_n_classes = dataset.num_classes
    data = dataset[0]
    data.y = data.y.squeeze()

    # the split idx is global idx
    train_idx = np.loadtxt('output/{}_train_idx.txt'.format(dataname)).astype(np.int32)
    valid_idx = np.loadtxt('output/{}_valid_idx.txt'.format(dataname)).astype(np.int32)
    test_idx = np.loadtxt('output/{}_test_idx.txt'.format(dataname)).astype(np.int32)
    real_classes = np.loadtxt('output/{}_real_classes.txt'.format(dataname)).astype(np.int8)
    n_classes = len(real_classes)
    data.held_mask = torch.zeros(len(data.y)).bool()
    data.held_mask[valid_idx] = True
    data.used_mask = torch.zeros(len(data.y)).bool()
    data.used_mask[train_idx] = True
    data.used_mask[valid_idx] = True
    data.used_mask[test_idx] = True

    # re-align real classes idx
    real_classes2idx = dict(zip(real_classes, [i for i in range(n_classes)]))
    for idx in train_idx:
        data.y[idx] = real_classes2idx[data.y[idx].item()]
    for idx in valid_idx:
        data.y[idx] = real_classes2idx[data.y[idx].item()]
    for idx in test_idx:
        data.y[idx] = real_classes2idx[data.y[idx].item()]
    data.y[~data.used_mask] = 0

    # Step 1: train GNN and record the sm vectors
    # gcn_res: output is probability-like
    pred = np.zeros((len(data.y), n_classes))
    pred[data.used_mask] = np.loadtxt('output/{}_oursplit_best_pred.txt'.format(dataname))  # for original training set without some rare classes
    pred = F.softmax(torch.tensor(pred), dim=1).numpy()
    print("pred shape: ", pred.shape)

    all_sm_vectors = np.zeros((10, len(data.y), n_classes))
    all_sm_vectors[:, data.used_mask] = np.load('output/{}_oursplit_all_sm_vectors.npy'.format(dataname))
    all_sm_vectors = F.softmax(torch.tensor(all_sm_vectors), dim=1).numpy()

    # Step 2: calculate confident joint and generate noise
    confident_joint = compute_confident_joint(data.y[valid_idx], pred[valid_idx], K=n_classes)
    print("Confident Joint: ", confident_joint)
    py, noise_matrix, inv_noise_matrix = estimate_latent(confident_joint, data.y[valid_idx])
    print("Noise Matrix (p(s|y)): ")
    print(noise_matrix)
    plt.figure()
    sns.heatmap(noise_matrix.T, cmap='PuBu', vmin=0, vmax=1, linewidth=1, annot=n_classes < 10)
    plt.title('Learned Noise Transition Matrix')
    plt.savefig('Case_Study_Noise_Matrix_' + dataname + '.jpg', bbox_inches='tight')
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
    labels = np.squeeze(data.y[test_idx].numpy())

    if dataname == 'ogbn-arxiv':
        mapping = pd.read_csv("./dataset/ogbn_{}/mapping/nodeidx2paperid.csv".format(dataname[5:]), index_col=0)
        catmap = pd.read_csv("./dataset/ogbn_{}/mapping/labelidx2arxivcategeory.csv".format(dataname[5:]), index_col=0)
    else:
        catmap = pd.read_csv("./dataset/ogbn_{}/mapping/labelidx2productcategory.csv".format(dataname[5:]), index_col=0)

    cert = pred[np.arange(len(labels)), predmax]
    cert_correct = pred[np.arange(len(labels)), labels]
    sort_idx = np.argsort(-probs)
    idx2real_classes = dict(zip([i for i in range(n_classes)], real_classes))

    for i in range(200):
        t_idx = sort_idx[i]  # test set indices
        ogb_idx = test_idx[t_idx]  # indices in ogb dataset change!!!
        cur_pred = predmax[t_idx]
        cur_label = labels[t_idx]
        if dataname == 'ogbn-arxiv':
            print("{} (test {} ogb {} arxiv {}): pred = {}[{}] ({:.5f}) actual = {}[{}] ({:.5f}) mislabel prob = {}".format(
                i, t_idx, ogb_idx, mapping.loc[ogb_idx, 'paper id'], idx2real_classes[cur_pred],
                catmap.loc[idx2real_classes[cur_pred], 'arxiv category'], cert[t_idx], idx2real_classes[cur_label],
                catmap.loc[idx2real_classes[cur_label], 'arxiv category'], cert_correct[t_idx], probs[t_idx]))
        else:
            print("{} (test {} ogb {}): pred = {}[{}] ({:.5f}) actual = {}[{}] ({:.5f}) mislabel prob = {}".format(
                i, t_idx, ogb_idx, idx2real_classes[cur_pred], catmap.loc[idx2real_classes[cur_pred], 'product category'],
                cert[t_idx], idx2real_classes[cur_label], catmap.loc[idx2real_classes[cur_label], 'product category'],
                cert_correct[t_idx], probs[t_idx]))
