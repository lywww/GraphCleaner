import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path as osp
import time
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, JumpingKnowledge, SGConv
from torch_geometric.data import NeighborSampler
import torch_geometric.transforms as T
import copy
import torch_sparse
# from logger import Logger
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import ModuleList, Linear, BatchNorm1d, Identity
from torch.utils.data import DataLoader
import nni
from nni.utils import merge_parameter
import argparse
import os

from Utils import setup_seed
setup_seed(1119)


# os.environ['MKL_THREADING_LAYER'] = 'GNU'
# os.environ['MKL_SERVICE_FORCE_INTEL'] = q
parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--l1reg', type=float, default=0.00003)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--num_epochs_patience', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=80000)
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
# parser.add_argument('--run_id', type=str)
# parser.add_argument('--dataset_split', type=str)
parser.add_argument('--iters', type=int, default=5000)
parser.add_argument('--walk_len', type=int, default=11)
args = parser.parse_args()
tuner_params = nni.get_next_parameter()
args = merge_parameter(args, tuner_params)
data_name = args.dataset
print("data_name: ", data_name)
# 加载数据集
dataset = PygNodePropPredDataset(name=data_name, root='./dataset/', transform=T.ToSparseTensor())
print(dataset)
data = dataset[0]
print(data)
n_classes = dataset.num_classes
if data_name == 'ogbn-products':
    n_classes = 42

# 划分数据集
split_idx = dataset.get_idx_split()

# 定义评估器
evaluator = Evaluator(name=data_name)

original_train_idx = split_idx['train']
original_train_y = data.y[original_train_idx]
train_idx, val_idx, test_idx = torch.tensor([]), torch.tensor([]), torch.tensor([])
real_classes = []
for c in range(n_classes):
    idx = torch.nonzero(original_train_y == c)[:,0]
    numbers = len(idx)
    if numbers < 50:
        print("Class {} is ignored!".format(c))
        continue
    real_classes.append(c)
    perm = np.random.permutation(numbers)
    train = idx[perm[:math.ceil(0.8 * numbers)]]
    val = idx[perm[math.ceil(0.8 * numbers):math.ceil(0.9 * numbers)]]
    test = idx[perm[math.ceil(0.9 * numbers):]]
    print("Class {} has {} training samples, {} validation samples, {} test samples".format(c, len(train), len(val), len(test)))
    train_idx = torch.cat([train_idx, train], 0)
    val_idx = torch.cat([val_idx, val], 0)
    test_idx = torch.cat([test_idx, test], 0)
train_idx = original_train_idx[train_idx.long()]
valid_idx = original_train_idx[val_idx.long()]
test_idx = original_train_idx[test_idx.long()]
original_train_idx = torch.cat((train_idx, valid_idx, test_idx))
n_classes = len(real_classes)

train_idx = train_idx.long()
val_idx = val_idx.long()
test_idx = test_idx.long()
print("Our split -- train: {} valid: {} test: {}".format(len(train_idx), len(val_idx), len(test_idx)))
np.savetxt('output/{}_train_idx.txt'.format(data_name), train_idx.numpy())
np.savetxt('output/{}_valid_idx.txt'.format(data_name), val_idx.numpy())
np.savetxt('output/{}_test_idx.txt'.format(data_name), test_idx.numpy())
np.savetxt('output/{}_real_classes.txt'.format(data_name), real_classes)

# test_idx = split_idx['test']
# val_idx = split_idx['valid']
# row, col, value = data.adj_t.coo()

data.adj_t = data.adj_t.to_symmetric()  # 对称归一化
# hop_adj =  gcn_norm(data.adj_t, add_self_loops=True)
# hop_adj = hop_adj @ hop_adj
# row, col, value = hop_adj.coo()
print('diag :', torch_sparse.get_diag(data.adj_t))
# print('samle adj on 0 :',data.adj_t.sample_adj(torch.LongTensor(0), 10))
# print('samle adj on 0 :',torch_sparse.index_select(data.adj_t,0 , torch.LongTensor(1)))
train_set = set(train_idx)
walk_length = args.walk_len
print('train idx:', train_idx)
print('val idx:', val_idx)
print('test idx:', test_idx)
print('original_train_idx: ', original_train_idx)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

node_size = data.x.shape[0]

lb = copy.deepcopy(data.y.squeeze(1))
lb[val_idx] = dataset.num_classes
lb[test_idx] = dataset.num_classes
iters = args.iters
print('iters :', iters)
print('lb shape:', lb.shape)
train_size = train_idx.shape[0]
t = time.perf_counter()
path = f'./preprocessed/{data_name[5:]}_rwarr_length{walk_length}.pt'
if osp.exists(path):
    label_emb = torch.load(path)
else:

    label_emb = torch.zeros([node_size, walk_length - 1, dataset.num_classes + 1]).float()
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
        # lastidx = rw[:,walk_length-1]
        # cond1 =  torch.eq(lastidx, train_idx)
        # lastidxlb = lb[lastidx]
        # label_emb[train_idx, lastidxlb] += torch.where(cond1, 0 ,1).long()
        # for i in range(train_size):
        #     label_emb[train_idx][i, lb[lastidx[i]]] =
        # print('label_emb[train_idx, lastidxlb] shape:',label_emb[train_idx, lastidxlb].shape)
        # label_emb[train_idx].index_put_((torch.LongTensor(range(train_size)),lastidxlb),torch.where(cond1, 0 ,1).long(), accumulate = True)
        # selflb = lb[train_idx]
        # label_emb[train_idx][selflb] = 0
        print(f'{i}th iteration Done! [{time.perf_counter() - t:.2f}s]')
    for i in range(iters):
        rw = torch_sparse.random_walk(data.adj_t, test_idx, walk_length)
        # rw = random_walk(rowptr.long(), col.long(), test_idx, walk_length, coalesced = True, p = 1, q = 1, num_nodes =None)
        # print('rw ',rw)
        for j in range(1, walk_length):
            lastidx = rw[:, j]
            cond1 = torch.eq(lastidx, test_idx)
            lastidxlb = lb[lastidx]
            jth_label_emb = label_emb[:, j - 1, :]
            jth_label_emb[test_idx, lastidxlb.long()] += torch.where(cond1, 0, 1).long()
    # label_emb[test_idx][selflb] = 0

    for i in range(iters):
        rw = torch_sparse.random_walk(data.adj_t, val_idx, walk_length)
        # rw = random_walk(rowptr.long(), col.long(), val_idx, walk_length, coalesced = True, p = 1, q = 1, num_nodes =None)
        # print('rw ',rw)
        for j in range(1, walk_length):
            lastidx = rw[:, j]
            cond1 = torch.eq(lastidx, val_idx)
            lastidxlb = lb[lastidx]
            jth_label_emb = label_emb[:, j - 1, :]
            jth_label_emb[val_idx, lastidxlb.long()] += torch.where(cond1, 0, 1).long()
    # label_emb[val_idx][selflb] = 0

    label_emb = label_emb / iters
    label_emb = label_emb[:, :, real_classes]
    # label_emb = F.softmax(label_emb, dim = 2)
    print('label_emb shape:', label_emb.shape)
    print('dataset.num_classes:', n_classes)
    torch.save(label_emb.cpu(), path)


# 定义网络
# GCN
class GCNNet(nn.Module):
    def __init__(self, dataset, hidden=256, num_layers=3):
        """
        :param dataset: 数据集
        :param hidden: 隐藏层维度，默认256
        :param num_layers: 模型层数，默认为3
        """
        super(GCNNet, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(dataset.num_node_features, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))

        for i in range(self.num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.convs.append(GCNConv(hidden, n_classes))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)  # 小数据集不norm反而效果更好
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, adj_t)
        x = F.log_softmax(x, dim=1)

        return x


class mixer(torch.nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int, L=10):
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
    # GCN_res


class GCN_res(nn.Module):
    def __init__(self, dataset, hidden=256, num_layers=6):
        super(GCN_res, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # self.input_fc = nn.Linear(dataset.num_node_features, hidden)

        # for i in range(self.num_layers):
        #     self.convs.append(GCNConv(hidden, hidden))
        #     self.bns.append(nn.BatchNorm1d(hidden))

        # self.out_fc = nn.Linear(100, dataset.num_classes)
        self.out_fc1 = nn.Linear(2 * hidden, n_classes)
        self.out_fc2 = nn.Linear(n_classes, n_classes)
        # self.weights = torch.nn.Parameter(torch.randn((len(self.convs) + 1)) )
        # self.lb_conv = SGConv(dataset.num_classes, hidden, K=0)
        # self.lb_conv = nn.Linear(dataset.num_classes, hidden)
        self.mixer1 = mixer(hidden * 2, dataset.num_node_features + n_classes)
        self.mixer12 = mixer(hidden * 2, dataset.num_node_features + n_classes)
        # self.mixer2 = mixer(dataset.num_node_features + dataset.num_classes, dataset.num_node_features + dataset.num_classes)
        # self.mixer2 = mixer(dataset.num_classes, dataset.num_node_features + dataset.num_classes, L = 50)
        # self.mixer22 = mixer(dataset.num_classes, dataset.num_node_features + dataset.num_classes, L = 50)
        self.conv1 = nn.Conv1d(dataset.num_node_features, hidden, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.conv2 = nn.Conv1d(n_classes, hidden, kernel_size=3, stride=1,
                               padding=1, bias=True)
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


# 实例化模型
# model = GCNNet(dataset=dataset, hidden=256, num_layers=3)
model = GCN_res(dataset=dataset, hidden=args.hidden, num_layers=8)
print(model)

# 转换为cpu或cuda格式
print(device)
model.to(device)
# data = data.to(device)

data.adj_t = data.adj_t.to_symmetric()  # 对称归一化
train_idx = train_idx.to(device)
test_idx = test_idx.to(device)
val_idx = val_idx.to(device)
print('train_idx shape :', train_idx.shape)
# 定义损失函数和优化器
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
label_emb = label_emb.to(device)

# 定义训练函数
x, adj_t = data.x, data.adj_t
xarr_path = f'./preprocessed/{data_name[5:]}_xarr_length{walk_length}.pt'
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


#     return loss
def L1reg(model):
    lamda = args.l1reg
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(abs(param))
    return lamda * regularization_loss


def train(batch_size):
    model.train()
    # print('rwlb_train shape',rwlb_train.shape)
    y_train = data.y.squeeze(1)[train_idx]
    y_map = dict(zip(real_classes, [i for i in range(len(real_classes))]))
    for k,v in y_map.items():
        y_train[y_train == k] = v
    x_train = x_arr[train_idx]
    smlb_train = label_emb[train_idx]
    pred = torch.zeros(y_train.size(0), n_classes)

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
        loss = criterion(out, trainlb)
        # print('homo loss:',homo)
        # print('criterion loss:',loss)
        loss = loss + L1reg(model)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * idx.numel()
    pred = pred.argmax(dim=-1)
    train_acc = evaluator.eval({'y_true': data.y[train_idx], 'y_pred': pred.unsqueeze(1)})['acc']
    return total_loss / y_train.size(0), train_acc


# def train_mixup():
#     model.train()
#     train_size = train_idx.shape[0]
#     np.random.beta(alpha, alpha, size=(train_size))

#     whole_idx = list(range(train_size))
#     random.shuffle(whole_idx)

#     out, layerout = model(data)
#     oriemb = out[train_idx]
#     orilb = data.y.squeeze(1)[train_idx]
#     trainlb = lam * data.y.squeeze(1)[train_idx] + (1 - lam)

#     np.random.dirichlet((1,1,1), (2))

#     loss = criterion(out[train_idx], data.y.squeeze(1)[train_idx])

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     return loss.item()

# 定义测试函数
@torch.no_grad()
def test(input_idx, batch_size):
    model.eval()
    # print('rwlb_train shape',rwlb_train.shape)
    y_train = data.y.squeeze(1)[input_idx]
    y_map = dict(zip(real_classes, [i for i in range(len(real_classes))]))
    for k, v in y_map.items():
        y_train[y_train == k] = v
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
        loss = criterion(out, lb)
        # loss.backward()
        # optimizer.step()

        total_loss += float(loss) * idx.numel()
    toppred = pred.argmax(dim=-1)
    train_acc = evaluator.eval({'y_true': data.y[input_idx], 'y_pred': toppred.unsqueeze(1)})['acc']
    return total_loss / y_train.size(0), train_acc, pred


# def test():
#     model.eval()

#     out = model(x_arr, label_emb)
#     y_pred = out.argmax(dim=-1, keepdim=True)

#     train_acc = evaluator.eval({
#         'y_true': data.y[split_idx['train']],
#         'y_pred': y_pred[split_idx['train']],
#     })['acc']
#     valid_acc = evaluator.eval({
#         'y_true': data.y[split_idx['valid']],
#         'y_pred': y_pred[split_idx['valid']],
#     })['acc']
#     test_acc = evaluator.eval({
#         'y_true': data.y[split_idx['test']],
#         'y_pred': y_pred[split_idx['test']],
#     })['acc']

#     return train_acc, valid_acc, test_acc


# 程序入口
if __name__ == '__main__':
    runs = 1
    # logger = Logger(runs)
    batch_size = args.batch_size
    best_test_acc = 0.
    best_valid_acc = 0.
    vacc_mx = 0.
    vlss_mn = 1e6
    all_smlb_vectors = []
    for run in range(runs):
        print(sum(p.numel() for p in model.parameters()))
        model.reset_parameters()

        for epoch in range(args.epochs):
            loss, train_acc = train(batch_size)
            # print('Epoch {:03d} train_loss: {:.4f}'.format(epoch, loss))

            valid_loss, valid_acc, valid_pred = test(val_idx, batch_size)
            test_loss, test_acc, test_pred = test(val_idx, batch_size)
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
                _, _, best_pred = test(original_train_idx, batch_size)
            if (epoch + 1) % 20 == 0:
                _, _, all_pred = test(original_train_idx, batch_size)
                all_smlb_vectors.append(all_pred.numpy())
        np.savetxt('output/{}_oursplit_best_pred.txt'.format(data_name), best_pred.numpy(), fmt='%.4f')
        np.save('output/{}_oursplit_all_sm_vectors.npy'.format(data_name), np.array(all_smlb_vectors))

        # logger.add_result(run, result)
    nni.report_final_result(best_test_acc)
    # logger.print_statistics()
