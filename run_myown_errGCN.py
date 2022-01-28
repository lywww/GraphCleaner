import os
import json
import argparse
import random
import numpy as np
import pandas as pd

import torch
from torch_geometric.datasets import Flickr
from torch_geometric.nn.conv import GCNConv
import torch.nn.functional as F

from run_GNNs import train_GNNs
from evaluate_different_methods import cal_patk, get_ytest


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


def get_data(dataset, noise_type, mislabel_rate, psx):
    if dataset == 'Flickr':
        dir = os.path.join('./dataset', 'Flickr')
        flickr = Flickr(dir)
        data = flickr[0]
        with open(os.path.join(dir, 'raw', 'noisy_class_map_'+noise_type+'_'+str(mislabel_rate)+'.json'), 'r') as f:
            noisy_class_map = json.load(f)
        for i in range(len(noisy_class_map)):
            data.y[i] = noisy_class_map[str(i)]
        data.x = torch.from_numpy(psx).float()
        for i,c in noisy_class_map.items():
            data.x[int(i)][c] -= 1
        return data, psx.shape[1]


class myGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)


def myown(psx, dataset, noise_type, mislabel_rate, mislabel_result_file):
    # prepare for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    data, n_features = get_data(dataset, noise_type, mislabel_rate, psx)
    N = data.num_nodes
    adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(len(data.edge_index[0])), (N, N)).to_dense()
    print("adj: ", adj)
    print("adj[0]: ", adj[0])
    data.to(device)
    print("errGCNData: ", data)
    model = myGCN(in_channels=n_features, hidden_channels=n_features*5, out_channels=n_features)
    model.to(device)
    print("errGCNModel: ", model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)  # lr and wd require tunning

    # load / train the model
    model.train()
    for epoch in range(500):  # n_epochs requires tunning
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out[data.train_mask], data.x[data.train_mask])
        print("mse loss: {:.2f}".format(loss))
        loss *= len(data.train_mask)
        for idx in data.train_mask:
            t = adj[idx]
            adjs = torch.nonzero(t).squeeze()
            print("adjs: ", adjs)
            adj_x = data.x[adjs]
            print("adj_x: ", adj_x)
            cur_out = torch.stack([out[idx] for i in range(len(adjs))], dim=0)
            loss += F.mse_loss(cur_out, adj_x)
            print("adj loss {:.2f}".format(F.mse_loss(cur_out, adj_x)))
        loss /= len(data.train_mask)
        print("Epoch[{}] Loss: {:.2f}".format(epoch + 1, loss))
        loss.backward()
        optimizer.step()

    # evaluate on validation set
    model.eval()
    predictions = model(data)
    predictions = predictions[data.train_mask]

    # get the result
    result_file = mislabel_result_file + '.csv'
    error = torch.norm(predictions, p=1, dim=1)
    idx2error = dict(zip([i for i in range(len(error))], error))
    er = [x[0] for x in sorted(idx2error.items(), key=lambda x: x[1], reverse=True)]

    # Save the results (True means wrong label)
    cl_results = pd.DataFrame({'ordered_errors': er})
    cl_results.to_csv(result_file, index=False)
    print("errGCN results saved!")
    cal_patk(er, get_ytest(dataset, noise_type, mislabel_rate))
    return er


if __name__ == "__main__":
    setup_seed(1119)

    parser = argparse.ArgumentParser(description="Generate Noises")
    parser.add_argument("--dataset", type=str, default='Flickr')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--mislabel_rate", type=float, default=0.1)
    parser.add_argument("--noise_type", type=str, default='symmetric')
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    args = parser.parse_args()

    ensure_dir('tensorboard_logs')
    log_dir = 'tensorboard_logs/{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format\
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
    ensure_dir('checkpoints')
    trained_model_file = 'checkpoints/{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format\
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
    ensure_dir('gnn_results')
    gnn_result_file = 'gnn_results/{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format\
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
    ensure_dir('mislabel_results')
    mislabel_result_file = 'mislabel_results/errGCN-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format \
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)

    # get the prediction results and save to file
    predictions, noisy_y = train_GNNs(args.model, args.dataset, args.noise_type, args.mislabel_rate, args.n_epochs,
                                      args.lr, args.weight_decay, log_dir, trained_model_file, 'myown')
    result = pd.DataFrame(data=np.hstack((predictions, noisy_y.reshape((-1,1)))))
    result.to_csv(gnn_result_file+'.csv', index=False, header=None)
    print("{} results saved!".format(args.model))

    # get noise indices
    myown(predictions, args.dataset, args.noise_type, args.mislabel_rate, mislabel_result_file)
