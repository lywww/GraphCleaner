import os
import json
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from reliability_diagrams import reliability_diagram

import torch
from torch_geometric.datasets import Flickr
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv
from torch_geometric.nn.models import GIN
import torch.nn.functional as F

from aum import AUMCalculator

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data(dataset, noise_type, mislabel_rate, special_set=None):
    if dataset == 'Flickr':
        dir = os.path.join('./dataset', 'Flickr')
        flickr = Flickr(dir)
        n_classes = flickr.num_classes
        data = flickr[0]
        with open(os.path.join(dir, 'raw', 'noisy_class_map_'+noise_type+'_'+str(mislabel_rate)+'.json'), 'r') as f:
            noisy_class_map = json.load(f)
        noisy_map = np.zeros(len(noisy_class_map), dtype=bool)
        for i in range(len(noisy_class_map)):
            if data.y[i] != noisy_class_map[str(i)]:
                data.y[i] = noisy_class_map[str(i)]
                noisy_map[i] = True
        if special_set and special_set[:3] == 'AUM':
            threshold_samples = pd.read_csv('./aum_data/' + dataset + '_aum_threshold_samples.csv')
            if special_set[-1] == '1':
                threshold_idx = threshold_samples['first_threshold_samples']
            else:
                threshold_idx = threshold_samples['second_threshold_samples']
            for idx in threshold_idx:
                data.y[idx] = n_classes
            n_classes += 1

        return data, n_classes, noisy_map[data.train_mask]


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

    def get_logits(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x

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

    def get_logits(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gin(x, edge_index)
        return x


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

    def get_logits(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x


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
    data, n_classes, noisy_map = get_data(dataset, noise_type, mislabel_rate, sepcial_set)
    data.to(device)
    n_features = data.num_features
    writer = create_summary_writer(log_dir)
    print("Data: ", data)

    # prepare model
    if model_name == 'GCN':
        model = GCN(in_channels=n_features, hidden_channels=250, out_channels=n_classes)
    elif model_name == 'GIN':
        model = myGIN(in_channels=n_features, hidden_channels=250, out_channels=n_classes)
    elif model_name == 'GAT':
        model = GAT(in_channels=n_features, hidden_channels=250, out_channels=n_classes)
    model.to(device)
    print("Model: ", model)

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # load / train the model
    clean_pos_logits, clean_sec_logits = [], []  # sample 5 for sym0.1
    noisy_pos_logits, noisy_sec_logits = [], []  # sample 107 for sym0.1
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
        if epoch == 99:
            # draw loss distribution
            samplewise_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], reduction='none')
            samplewise_loss = samplewise_loss.cpu().detach().numpy()
            incorrect_loss = preprocessing.normalize([samplewise_loss[noisy_map]])
            correct_loss = preprocessing.normalize([samplewise_loss[~noisy_map]])
            print("incorrect loss: ", incorrect_loss)
            print("correct loss: ", correct_loss)
            plt.hist(incorrect_loss[0], bins=50, label='incorrect label', alpha=0.35)
            plt.hist(correct_loss[0], bins=50, label='correct label', alpha=0.35)
            plt.legend(loc='upper left')
            plt.title(model_name+' loss distribution')
            plt.savefig(model_name+'_'+noise_type+'_'+str(mislabel_rate)+'_epoch100_loss_dist.jpg', bbox_inches='tight')
            plt.show()
            # draw softmax distribution
            # softmax = out[data.train_mask].cpu().detach().numpy()
            # incorrect_softmax = preprocessing.normalize(softmax[noisy_map])
            # correct_softmax = preprocessing.normalize(softmax[~noisy_map])
            # print("incorrect softmax: ", incorrect_softmax)
            # print("correct softmax: ", correct_softmax)
            # plt.hist(incorrect_softmax[0], bins=10, label='incorrect softmax', alpha=0.35)
            # plt.hist(correct_softmax[0], bins=10, label='correct softmax', alpha=0.35)
            # plt.legend(loc='upper left')
            # plt.show()
        logits = model.get_logits(data).cpu().detach().numpy()
        clean_logits = logits[5]
        noisy_logits = logits[107]
        clean_pos_logits.append(clean_logits[6])
        noisy_pos_logits.append(noisy_logits[1])
        clean_logits = np.delete(clean_logits, [6])
        noisy_logits = np.delete(noisy_logits, [1])
        clean_sec_logits.append(max(clean_logits))
        noisy_sec_logits.append(max(noisy_logits))
        if sepcial_set and sepcial_set[:3] == 'AUM':
            aum_input = out[data.train_mask]
            aum_input = to_softmax(aum_input.cpu().detach().numpy())
            aum_calculator.update(torch.from_numpy(aum_input).to(device), data.y[data.train_mask], sample_ids)
    torch.save(model.state_dict(), trained_model_file)
    if sepcial_set and sepcial_set[:3] == 'AUM':
        aum_calculator.finalize()

    # draw logits
    x = [i for i in range(200)]
    plt.plot(x, clean_pos_logits, 'g--', label='logit')
    plt.plot(x, clean_sec_logits, 'r--', label='largest other logit')
    plt.title('Correctly Labelled')
    plt.xlabel('Training Epoch')
    plt.ylabel('Logit Value')
    plt.legend()
    plt.savefig(model_name + '_' + noise_type + '_' + str(mislabel_rate) + '_clean_logits.jpg',
                bbox_inches='tight')
    plt.show()

    plt.plot(x, noisy_pos_logits, 'g--', label='logit')
    plt.plot(x, noisy_sec_logits, 'r--', label='largest other logit')
    plt.title('Mislabeled')
    plt.xlabel('Training Epoch')
    plt.ylabel('Logit Value')
    plt.legend()
    plt.savefig(model_name + '_' + noise_type + '_' + str(mislabel_rate) + '_noisy_logits.jpg',
                bbox_inches='tight')
    plt.show()

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


if __name__ == "__main__":
    setup_seed(1119)

    parser = argparse.ArgumentParser(description="Run GNNs for Drawing")
    parser.add_argument("--dataset", type=str, default='Flickr')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--mislabel_rate", type=float, default=0.1)
    parser.add_argument("--noise_type", type=str, default='symmetric')
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    args = parser.parse_args()

    log_dir = 'tensorboard_logs/{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format\
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
    trained_model_file = 'checkpoints/{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format\
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
    gnn_result_file = 'gnn_results/{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format\
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
    mislabel_result_file = 'mislabel_results/CL-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format \
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)

    # get the prediction results and save to file
    predictions, noisy_y = train_GNNs(args.model, args.dataset, args.noise_type, args.mislabel_rate, args.n_epochs,
                                      args.lr, args.weight_decay, log_dir, trained_model_file)
