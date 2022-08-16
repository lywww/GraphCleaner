import numpy as np
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path as osp
import time
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric
from torch_geometric.data import NeighborSampler, Data
from torch_geometric.utils import subgraph, remove_isolated_nodes
import torch_geometric.transforms as T
import copy
import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch.utils.data import DataLoader
import nni
from nni.utils import merge_parameter
import argparse

from Utils import setup_seed

setup_seed(1119)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class mixer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, L=10):
        super(mixer, self).__init__()

        self.dropout1 = nn.Dropout(args.dropout_rate)
        self.dropout2 = nn.Dropout(args.dropout_rate)
        self.ln11 = Linear(L, L)
        self.ln12 = Linear(L, L)
        self.bn1 = BatchNorm1d(in_channels)

        self.ln21 = Linear(in_channels, in_channels)
        self.ln22 = Linear(in_channels, in_channels)
        self.bn2 = BatchNorm1d(L)
        # self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=10, stride=1,
        #              padding=0, bias=True)

        self.bn3 = BatchNorm1d(in_channels)

        # self.finalfc = Linear(in_channels , out_channels)

    def reset_parameters(self):
        self.ln11.reset_parameters()
        self.ln12.reset_parameters()
        self.ln21.reset_parameters()
        self.ln12.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

    def forward(self, x):
        # print('x_smlb shape',x_smlb.shape)
        x = x.transpose(1, 2)
        skipx = x
        x = self.ln11(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.ln12(x)
        x += skipx
        x = self.dropout1(x)

        x = x.transpose(1, 2)
        skipx = x
        x = self.ln21(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.ln22(x)
        x += skipx
        x = self.dropout2(x)
        return x


class GCN_res(nn.Module):
    def __init__(self, num_classes, num_node_features, hidden=256, num_layers=6):
        super(GCN_res, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # self.input_fc = nn.Linear(dataset.num_node_features, hidden)

        # for i in range(self.num_layers):
        #     self.convs.append(GCNConv(hidden, hidden))
        #     self.bns.append(nn.BatchNorm1d(hidden))

        # self.out_fc = nn.Linear(100, dataset.num_classes)
        self.out_fc1 = nn.Linear(2 * hidden, num_classes)
        self.out_fc2 = nn.Linear(num_classes, num_classes)
        # self.weights = torch.nn.Parameter(torch.randn((len(self.convs) + 1)) )
        # self.lb_conv = SGConv(dataset.num_classes, hidden, K=0)
        # self.lb_conv = nn.Linear(dataset.num_classes, hidden)
        self.mixer1 = mixer(hidden * 2, num_node_features + num_classes)
        self.mixer12 = mixer(hidden * 2, num_node_features + num_classes)
        # self.mixer2 = mixer(dataset.num_node_features + dataset.num_classes, dataset.num_node_features + dataset.num_classes)
        # self.mixer2 = mixer(dataset.num_classes, dataset.num_node_features + dataset.num_classes, L = 50)
        # self.mixer22 = mixer(dataset.num_classes, dataset.num_node_features + dataset.num_classes, L = 50)
        self.conv1 = nn.Conv1d(num_node_features, hidden, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv1d(num_classes, hidden, kernel_size=3, stride=1, padding=1, bias=True)
        self.avgpool1 = nn.AvgPool1d(kernel_size=10)

    def reset_parameters(self):
        # for conv in self.convs:
        #     conv.reset_parameters()
        # for bn in self.bns:
        #     bn.reset_parameters()
        # self.input_fc.reset_parameters()
        self.out_fc1.reset_parameters()
        self.out_fc2.reset_parameters()
        # torch.nn.init.normal_(self.weights)

    def forward(self, data, label_emb):
        # print('data shape :',data.shape)
        # print('label_emb shape :',label_emb.shape)
        # print('x shape:',x.shape)
        data = data.transpose(1, 2)
        data = self.conv1(data)
        data = data.transpose(1, 2)

        label_emb = label_emb.transpose(1, 2)
        label_emb = self.conv2(label_emb)
        label_emb = label_emb.transpose(1, 2)

        # print('data shape :',data.shape)
        # print('label_emb shape :',label_emb.shape)
        x = torch.cat((data, label_emb), dim=2)

        x = self.mixer1(x)
        x = self.mixer12(x)
        # label_emb = self.mixer2(label_emb)
        # label_emb = self.mixer22(label_emb)
        # x += label_emb
        x = x.transpose(1, 2)

        # x = self.conv1(1)
        # x = F.relu(x)
        x = self.avgpool1(x)

        x = x.view(x.shape[0], -1)
        emb = x
        # x = self.out_fc(x)
        x = self.out_fc1(x)

        x = F.log_softmax(x, dim=1)
        # print('len: ',len(layer_out))
        return x, emb


def randomWalk(args, data, n_classes, n_nodes):
    data.adj_t = data.adj_t.to_symmetric()  # 对称归一化
    node_size = n_nodes
    walk_length = args.walk_len
    data_name = args.dataset
    train_idx = torch.nonzero(data.train_mask == True)[:,0]
    val_idx = torch.nonzero(data.val_mask == True)[:, 0]
    test_idx = torch.nonzero(data.test_mask == True)[:, 0]
    print('train idx:', train_idx)
    print('val idx:', val_idx)
    print('test idx:', test_idx)

    lb = copy.deepcopy(data.y.squeeze(1))
    lb[val_idx] = n_classes
    lb[test_idx] = n_classes
    iters = args.iters
    print('iters :', iters)
    print('lb shape:', lb.shape)
    t = time.perf_counter()
    path = f'./preprocessed/{data_name[5:]}_rwarr_length{walk_length}.pt'
    if osp.exists(path):
        label_emb = torch.load(path)
    else:

        label_emb = torch.zeros([node_size, walk_length - 1, n_classes + 1]).float()
        data.adj_t = data.adj_t.to(device)
        train_idx = train_idx.to(device)
        test_idx = test_idx.to(device)
        val_idx = val_idx.to(device)
        label_emb = label_emb.to(device)
        for i in range(iters):
            rw = torch_sparse.random_walk(data.adj_t, train_idx, walk_length)
            # rw = random_walk(rowptr.long(), col.long(), train_idx, walk_length, coalesced = True, p = 1, q = 1, num_nodes =None)
            # print('rw ',rw)
            for j in range(1, walk_length):
                lastidx = rw[:, j]
                cond1 = torch.eq(lastidx, train_idx)
                lastidxlb = lb[lastidx]
                jth_label_emb = label_emb[:, j - 1, :]
                jth_label_emb[train_idx, lastidxlb.long()] += torch.where(cond1, 0, 1).long()
            print(f'{i}th iteration Done! [{time.perf_counter() - t:.2f}s]')
        for i in range(iters):
            rw = torch_sparse.random_walk(data.adj_t, test_idx, walk_length)
            for j in range(1, walk_length):
                lastidx = rw[:, j]
                cond1 = torch.eq(lastidx, test_idx)
                lastidxlb = lb[lastidx]
                jth_label_emb = label_emb[:, j - 1, :]
                jth_label_emb[test_idx, lastidxlb.long()] += torch.where(cond1, 0, 1).long()

        for i in range(iters):
            rw = torch_sparse.random_walk(data.adj_t, val_idx, walk_length)
            for j in range(1, walk_length):
                lastidx = rw[:, j]
                cond1 = torch.eq(lastidx, val_idx)
                lastidxlb = lb[lastidx]
                jth_label_emb = label_emb[:, j - 1, :]
                jth_label_emb[val_idx, lastidxlb.long()] += torch.where(cond1, 0, 1).long()

        label_emb = label_emb / iters
        label_emb = label_emb[:, :, :n_classes]
        print('label_emb shape:', label_emb.shape)
        torch.save(label_emb.cpu(), path)

    return label_emb, train_idx, val_idx, test_idx


def sampleSubGraph():
    print("Loading dataset......")
    dataset = PygNodePropPredDataset(name='ogbn-papers100M', root='./dataset/')
    data = dataset[0]
    print(dataset)

    print("Sampling subdataset......")
    split_idx = dataset.get_idx_split()
    # test_idx = split_idx['test']
    # testy = data.y[test_idx]
    test_idx = split_idx['train'][:200000]
    testy = data.y[test_idx]
    eliminate_idx = []
    for c in range(dataset.num_classes):
        if 0 < sum(testy == c) < 1000:
            idx = test_idx[torch.nonzero(testy == c)[:, 0]]
            for i in idx:
                eliminate_idx.append(int(i))
            print("Class {} with {} nodes is eliminated!".format(c, len(idx)))
    eliminate_idx = torch.tensor(eliminate_idx)
    real_test_idx = torch.tensor([i for i in test_idx if i not in eliminate_idx])
    real_test_idx = real_test_idx[:-2]
    print("There are {} nodes to eliminate and 2 raw isolated nodes, resulting in {} nodes to use".format(len(eliminate_idx), len(real_test_idx)))
    edge_index, _ = subgraph(subset=real_test_idx, edge_index=data.edge_index, relabel_nodes=True)
    # _, _, mask = remove_isolated_nodes(edge_index)
    # print("Remove {} isolated nodes".format(sum(mask)))
    # real_test_idx = real_test_idx[mask]
    #
    # edge_index, _ = subgraph(subset=real_test_idx, edge_index=data.edge_index, relabel_nodes=True)
    x = data.x[real_test_idx]
    y = data.y[real_test_idx]
    idx2ogbidx = {}
    idx2testidx = {}
    for i, idx in enumerate(real_test_idx):
        idx2ogbidx[i] = int(idx)
        idx2testidx[i] = int(torch.nonzero(test_idx==idx)[0])
    with open('./output/idx2ogbidx_train', 'wb') as file:
        pickle.dump(idx2ogbidx, file)
    with open('./output/idx2testidx_train', 'wb') as file:
        pickle.dump(idx2testidx, file)

    print("Generating Data......")
    unique_y = torch.unique(y)
    oldy2newy = dict()
    next_new_y = 0
    for i in unique_y:
        oldy2newy[int(i)] = next_new_y
        next_new_y += 1
    with open('./output/oldy2newy_train', 'wb') as file:
        pickle.dump(oldy2newy, file)
    for i, cur_y in enumerate(y):
        y[i] = oldy2newy[int(cur_y)]
    data = Data(x=x, y=y, edge_index=edge_index)
    n_classes = len(oldy2newy)
    n_nodes = len(real_test_idx)
    data.adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(n_nodes, n_nodes))
    perm = np.random.permutation(n_nodes)
    data.train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.train_mask[perm[:math.ceil(0.8 * n_nodes)]] = True
    data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.val_mask[perm[math.ceil(0.8 * n_nodes):math.ceil(0.9 * n_nodes)]] = True
    data.test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.test_mask[perm[math.ceil(0.9 * n_nodes):]] = True

    return data, n_nodes, n_classes


def L1reg(args, model):
    lamda = args.l1reg
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(abs(param))
    return lamda * regularization_loss


def train(args, model, criterion, optimizer, batch_size, evaluator, x_train, y_train, smlb_train, pred):
    model.train()
    total_loss = 0
    alpha = 0.2
    for idx in DataLoader(range(y_train.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        input = x_train[idx].to(device)
        input_smlb = smlb_train[idx].to(device)
        trainlb = y_train[idx].to(device)
        out, emb = model(input, input_smlb)
        pred[idx] = out.detach().cpu()
        # homo = contra_loss(emb, idx)
        loss = criterion(out, trainlb.long())
        # print('homo loss:',homo)
        # print('criterion loss:',loss)
        loss = loss + L1reg(args, model)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * idx.numel()
    pred = pred.argmax(dim=-1)
    train_acc = evaluator.eval({'y_true': y_train.unsqueeze(1), 'y_pred': pred.unsqueeze(1)})['acc']
    return total_loss / y_train.size(0), train_acc


@torch.no_grad()
def test(model, criterion, data, n_classes, x_arr, label_emb, input_idx, batch_size, evaluator):
    model.eval()
    # print('rwlb_train shape',rwlb_train.shape)
    y_train = data.y.squeeze(1)[input_idx]
    x_train = x_arr[input_idx]
    smlb_train = label_emb[input_idx]
    pred = torch.zeros(y_train.size(0), n_classes)

    total_loss = 0
    for idx in DataLoader(range(y_train.size(0)), batch_size, shuffle=True):
        # optimizer.zero_grad()
        input = x_train[idx].to(device)
        input_smlb = smlb_train[idx].to(device)
        lb = y_train[idx].to(device)
        out, emb = model(input, input_smlb)
        pred[idx] = out.detach().cpu()
        loss = criterion(out, lb.long())
        # loss.backward()
        # optimizer.step()

        total_loss += float(loss) * idx.numel()
    toppred = pred.argmax(dim=-1)
    train_acc = evaluator.eval({'y_true': data.y[input_idx], 'y_pred': toppred.unsqueeze(1)})['acc']
    return total_loss / y_train.size(0), train_acc, pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--l1reg', type=float, default=0.00003)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_epochs_patience', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=80000)
    parser.add_argument('--dataset', type=str, default='ogbn-papers100M')
    parser.add_argument('--iters', type=int, default=5000)
    parser.add_argument('--walk_len', type=int, default=11)
    args = parser.parse_args()
    data_name = args.dataset
    evaluator = Evaluator(name=data_name)
    tuner_params = nni.get_next_parameter()
    args = merge_parameter(args, tuner_params)

    data, n_nodes, n_classes = sampleSubGraph()
    label_emb, train_idx, val_idx, test_idx = randomWalk(args, data, n_classes, n_nodes)

    # 实例化模型
    model = GCN_res(n_classes, len(data.x[0]), hidden=args.hidden, num_layers=8)
    print(model)

    # 转换为cpu或cuda格式
    model.to(device)
    data = data.to(device)

    data.adj_t = data.adj_t.to_symmetric()  # 对称归一化
    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)
    val_idx = val_idx.to(device)
    all_idx = torch.tensor([i for i in range(n_nodes)])
    all_idx = all_idx.to(device)
    print('train_idx shape :', train_idx.shape)

    # 定义损失函数和优化器
    criterion = nn.NLLLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    label_emb = label_emb.to(device)

    # 定义训练函数
    x, adj_t = data.x, data.adj_t
    xarr_path = f'./preprocessed/{data_name[5:]}_xarr_length{args.walk_len}.pt'
    if osp.exists(xarr_path):
        x_arr = torch.load(xarr_path)
    else:
        xs = [x]
        adj_t = gcn_norm(adj_t, add_self_loops=True)
        for i in range(9):
            x = adj_t @ x
            xs.append(x)
        x_arr = torch.stack(xs, dim=1)
        torch.save(x_arr, xarr_path)

    print("Start to train MLPMixer......")
    runs = 1
    batch_size = args.batch_size
    best_test_acc = 0.
    best_valid_acc = 0.
    vacc_mx = 0.
    vlss_mn = 1e6
    all_smlb_vectors = []
    for run in range(runs):
        print(sum(p.numel() for p in model.parameters()))
        model.reset_parameters()

        y_train = data.y.squeeze(1)[train_idx]
        x_train = x_arr[train_idx]
        smlb_train = label_emb[train_idx]
        pred = torch.zeros(y_train.size(0), n_classes)

        for epoch in range(args.epochs):
            loss, train_acc = train(args, model, criterion, optimizer, args.batch_size, evaluator, x_train, y_train, smlb_train, pred)
            # print('Epoch {:03d} train_loss: {:.4f}'.format(epoch, loss))

            valid_loss, valid_acc, valid_pred = test(model, criterion, data, n_classes, x_arr, label_emb, val_idx, batch_size, evaluator)
            test_loss, test_acc, test_pred = test(model, criterion, data, n_classes, x_arr, label_emb, test_idx, batch_size, evaluator)
            nni.report_intermediate_result(test_acc)
            # result = test(test_idx)
            # train_acc, valid_acc, test_acc = result
            result = (train_acc, valid_acc, test_acc)
            if valid_acc >= vacc_mx or valid_loss <= vlss_mn:
                vacc_mx = max((valid_acc, vacc_mx))
                vlss_mn = min((valid_loss, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= args.num_epochs_patience:
                    break
            best_test_acc = max(best_test_acc, test_acc)
            best_valid_acc = max(best_valid_acc, valid_acc)
            # print(f'Train: {train_acc:.4f}, Val: {valid_acc:.4f}, 'f'Test: {test_acc:.4f}')
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')
            if valid_acc == best_valid_acc:
                print("best valid acc")
                _, _, best_pred = test(model, criterion, data, n_classes, x_arr, label_emb, all_idx, batch_size, evaluator)
            if (epoch + 1) % 20 == 0:
                _, _, all_pred = test(model, criterion, data, n_classes, x_arr, label_emb, all_idx, batch_size, evaluator)
                all_smlb_vectors.append(all_pred.numpy())
        np.savetxt('output/{}_best_pred.txt'.format(data_name), best_pred.numpy(), fmt='%.4f')
        np.save('output/{}_all_sm_vectors.npy'.format(data_name), np.array(all_smlb_vectors))

    nni.report_final_result(best_test_acc)
