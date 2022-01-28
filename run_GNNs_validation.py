import os
import json
import pandas as pd
import numpy as np
from reliability_diagrams import reliability_diagram
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

import torch
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv
from torch_geometric.nn.models import GIN
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from aum import AUMCalculator
from run_GNNs import get_data

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


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


def draw_reliability_diagram(predictions, y, model_name, noise_type, mislabel_rate):
    predictions = to_softmax(predictions.cpu().detach().numpy())
    y = y.cpu().detach().numpy()
    fig = reliability_diagram(true_labels=y, pred_labels=np.argmax(predictions, axis=1),
                              confidences=np.max(predictions, axis=1), return_fig=True)
    fig.savefig('./' + model_name + '_' + noise_type + '_' + str(mislabel_rate) + '_reliability.jpg',
                bbox_inches='tight')


def train_GNNs(model_name, dataset, noise_type, mislabel_rate, n_epochs, lr, wd, log_dir, trained_model_file,
               special_set=None):
    # prepare noisy data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    data, n_classes = get_data(dataset, noise_type, mislabel_rate, special_set)
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
    scheduler = ReduceLROnPlateau(optimizer, mode='max')

    # load / train the model
    if (not special_set) and os.path.exists(trained_model_file):
        model.load_state_dict(torch.load(trained_model_file))
        model.eval()
        best_out = model(data)
    else:
        if special_set and special_set[:3] == 'AUM':
            aum_calculator = AUMCalculator('./aum_data', compressed=True)
            sample_ids = (data.test_mask == False).nonzero().reshape(-1)
            sample_ids = sample_ids.cpu().detach().numpy()
            val_ids = (data.val_mask == True).nonzero().reshape(-1)
            val_ids = val_ids.cpu().detach().numpy()
        best_cri = 0
        best_out = []
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            print("Epoch[{}] Loss: {:.2f}".format(epoch + 1, loss))
            writer.add_scalar("training/loss", loss, epoch)
            loss.backward()
            optimizer.step()
            if special_set and special_set[:3] == 'AUM':
                aum_input = out[~data.test_mask]
                aum_input = to_softmax(aum_input.cpu().detach().numpy())
                aum_calculator.update(torch.from_numpy(aum_input).to(device), data.y[~data.test_mask], sample_ids)

            model.eval()
            eval_out = model(data)
            y_pred = eval_out[data.val_mask].argmax(dim=-1).cpu().detach()
            y_true = data.y[data.val_mask].cpu().detach()
            cri = f1_score(y_true, y_pred, average='micro')
            if cri > best_cri:
                print("New Best Criterion: {:.2f}".format(cri))
                best_cri = cri
                best_out = eval_out
                torch.save(model.state_dict(), trained_model_file)
            scheduler.step(cri)
        if special_set and special_set[:3] == 'AUM':
            aum_calculator.finalize()

    # evaluate on validation set
    model.eval()
    predictions = best_out  # model(data)
    y = data.y
    tr_predictions = predictions[data.train_mask]
    tr_y = y[data.train_mask]
    draw_reliability_diagram(tr_predictions, tr_y, model_name, noise_type, mislabel_rate)
    if special_set != 'myown':
        predictions = predictions[data.val_mask]
        y = data.y[data.val_mask]

    predictions = to_softmax(predictions.cpu().detach().numpy())
    y = y.cpu().detach().numpy()
    if special_set and special_set[:3] == 'AUM':
        return predictions, y, val_ids
    if special_set and special_set[:5] == 'nbagg':
        return predictions, y, to_softmax(tr_predictions.cpu().detach().numpy()), tr_y.cpu().detach().numpy()
    return predictions, y
