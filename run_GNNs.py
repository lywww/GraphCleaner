import os
import json
import pandas as pd
import numpy as np
from reliability_diagrams import reliability_diagram

import torch
from torch_geometric.datasets import Flickr, Reddit2, Planetoid
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv
from torch_geometric.nn.models import GIN
import torch.nn.functional as F

from aum import AUMCalculator

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


def get_data(dataset, noise_type='symmetric', mislabel_rate=0, special_set=None):
    dir = os.path.join('./dataset', dataset)
    if dataset == 'Flickr' or dataset == 'Reddit2':  # tr, va, te are all corrupted
        if dataset == 'Flickr':
            data = Flickr(dir)
        elif dataset == 'Reddit2':
            data = Reddit2(dir)
        n_classes = data.num_classes
        data = data[0]

    elif dataset == 'Cora' or dataset == 'CiteSeer' or dataset == 'PubMed':
        data = Planetoid(root='./dataset', name=dataset)
        # train_mask = np.ones(len(data[0].train_mask), dtype=bool)
        # train_mask[data[0].val_mask] = False
        # train_mask[data[0].test_mask] = False
        n_classes = data.num_classes
        data = data[0]
        # data.train_mask = torch.from_numpy(train_mask)
        print("number of training samples in get_data: ", sum(data.train_mask))

    if mislabel_rate == 0:
        return data, n_classes

    with open(os.path.join(dir, 'raw', 'noisy_class_map_' + noise_type + '_' + str(mislabel_rate) + '.json'), 'r') as f:
        noisy_class_map = json.load(f)
    for i in range(len(noisy_class_map)):
        try:
            data.y[i] = noisy_class_map[str(i)]
        except:
            data.y[i] = noisy_class_map[i]
    if special_set and special_set[:3] == 'AUM':
        threshold_samples = pd.read_csv('./aum_data/' + dataset + '_aum_threshold_samples.csv')
        if special_set[-1] == '1':
            threshold_idx = threshold_samples['first_threshold_samples']
        else:
            threshold_idx = threshold_samples['second_threshold_samples']
        for idx in threshold_idx:
            data.y[idx] = n_classes
        n_classes += 1

    return data, n_classes


def create_summary_writer(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer


class GCN(torch.nn.Module):
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


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

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


class myGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gin = GIN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=2,
                       out_channels=out_channels, dropout=0.5)  # use the default dropout rate of F.dropout
        # default GIN has relu and dropout (because it uses MLP)
        # while the default GCN, GraphSage, GAT don't have relu and dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gin(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, out_channels)

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


def to_softmax(x):
    # transfer log_softmax to softmax
    max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp((x - max).astype(float))
    sum = np.sum(e_x, axis=1, keepdims=True)
    f_x = e_x / sum
    return f_x


def train_GNNs(model_name, dataset, noise_type, mislabel_rate, n_epochs, lr, wd, log_dir, trained_model_file,
               sepcial_set=None):
    # prepare noisy data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    data, n_classes = get_data(dataset, noise_type, mislabel_rate, sepcial_set)
    data.to(device)
    n_features = data.num_features
    writer = create_summary_writer(log_dir)
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

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # load / train the model
    if (not sepcial_set) and os.path.exists(trained_model_file):
        model.load_state_dict(torch.load(trained_model_file))
    else:
        if sepcial_set and sepcial_set[:3] == 'AUM':
            aum_calculator = AUMCalculator('./aum_data', compressed=True)
            sample_ids = (data.train_mask == True).nonzero().reshape(-1)
            sample_ids = sample_ids.cpu().detach().numpy()
        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            print("Epoch[{}] Loss: {:.2f}".format(epoch + 1, loss))
            writer.add_scalar("training/loss", loss, epoch)
            loss.backward()
            optimizer.step()
            if sepcial_set and sepcial_set[:3] == 'AUM':
                aum_input = out[data.train_mask]
                aum_input = to_softmax(aum_input.cpu().detach().numpy())
                aum_calculator.update(torch.from_numpy(aum_input).to(device), data.y[data.train_mask], sample_ids)
        torch.save(model.state_dict(), trained_model_file)
        if sepcial_set and sepcial_set[:3] == 'AUM':
            aum_calculator.finalize()

    # evaluate on validation set
    model.eval()
    predictions = model(data)
    y = data.y
    if sepcial_set != 'myown':
        predictions = predictions[data.train_mask]
        y = data.y[data.train_mask]
    predictions = to_softmax(predictions.cpu().detach().numpy())
    y = y.cpu().detach().numpy()
    fig = reliability_diagram(true_labels=y, pred_labels=np.argmax(predictions, axis=1),
                              confidences=np.max(predictions, axis=1), return_fig=True)
    fig.savefig('./'+model_name+'_'+noise_type+'_'+str(mislabel_rate)+'_reliability.jpg', bbox_inches='tight')
    return predictions, y
