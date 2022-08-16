import sys
import pickle
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch

from Utils import setup_seed
from case_study_baseclassifier import sampleSubGraph
from torch_geometric.utils import degree, to_undirected


if __name__ == '__main__':
    setup_seed(1119)
    dataname = 'ogbn-papers100M'

    origin = sys.stdout
    f = open('papers100_case_study_FSGNN_baseline_train.txt', 'w')
    sys.stdout = f

    # load data
    data, n_nodes, n_classes = sampleSubGraph()
    data.y = data.y.squeeze()
    train_idx = torch.nonzero(data.train_mask == True)[:, 0]
    val_idx = torch.nonzero(data.val_mask == True)[:, 0]
    test_idx = torch.nonzero(data.test_mask == True)[:, 0]

    # pred = np.loadtxt('output/{}_best_pred.txt'.format(dataname))
    # pred = F.softmax(torch.tensor(pred), dim=1).numpy()
    # print("pred shape: ", pred.shape)

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

    # Step 5: analysis
    pred = pred[test_idx]
    predmax = np.argmax(pred, axis=1)
    pred_probs = np.max(pred, axis=1)
    labels = [int(i) for i in data.y[test_idx]]

    cert = pred[np.arange(len(labels)), predmax]
    cert_correct = pred[np.arange(len(labels)), labels]
    probs = cert - cert_correct
    sort_idx = np.argsort(-probs)
    newy2oldy = dict(zip(oldy2newy.values(), oldy2newy.keys()))
    print("oldy2newy: ", oldy2newy)
    print("newy2oldy: ", newy2oldy)

    edge_index = to_undirected(data.edge_index)
    degree = degree(edge_index[0])
    for i in range(len(test_idx)):
        t_idx = sort_idx[i]  # test set indices
        tes_idx = int(test_idx[t_idx])
        if degree[tes_idx] >= 3:
            ogb_idx = idx2ogbidx[tes_idx]  # indices in ogb dataset change!!!
            cur_pred = predmax[t_idx]
            cur_label = labels[t_idx]
            if cur_pred == cur_label:
                continue
            print(
                "{} (test {} ogb {} arxiv {}): pred = {}[{}] ({:.5f}) actual = {}[{}] ({:.5f}) mislabel prob = {}".format(
                    i, t_idx, ogb_idx, mapping.loc[ogb_idx, 'paper id'], newy2oldy[cur_pred],
                    catmap.loc[newy2oldy[cur_pred], 'arxiv category'], cert[t_idx], newy2oldy[cur_label],
                    catmap.loc[newy2oldy[cur_label], 'arxiv category'], cert_correct[t_idx], probs[t_idx]))

    sys.stdout = origin
    f.close()
