import argparse
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from GNN_models import GCN, myGIN, GAT, baseMLP, myGraphUNet
from Utils import setup_seed, ensure_dir, get_data


def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().detach().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


def track_training_loss(device, all_losses):
    loss_tr = all_losses.cpu().detach().numpy()

    # outliers detection
    max_perc = np.percentile(loss_tr, 95)
    min_perc = np.percentile(loss_tr, 5)
    loss_tr = loss_tr[(loss_tr<=max_perc) & (loss_tr>=min_perc)]

    bmm_model_maxLoss = torch.FloatTensor([max_perc]).to(device)
    bmm_model_minLoss = torch.FloatTensor([min_perc]).to(device) + 10e-6


    loss_tr = (loss_tr - bmm_model_minLoss.data.cpu().numpy()) / (bmm_model_maxLoss.data.cpu().numpy() - bmm_model_minLoss.data.cpu().numpy() + 1e-6)

    loss_tr[loss_tr>=1] = 1-10e-4
    loss_tr[loss_tr <= 0] = 10e-4

    bmm_model = BetaMixture1D(max_iters=10)
    bmm_model.fit(loss_tr)

    bmm_model.create_lookup(1)

    return bmm_model, bmm_model_maxLoss, bmm_model_minLoss


def compute_probabilities_batch(batch_losses, bmm_model, bmm_model_maxLoss, bmm_model_minLoss):
    batch_losses = (batch_losses - bmm_model_minLoss) / (bmm_model_maxLoss - bmm_model_minLoss + 1e-6)
    batch_losses[batch_losses >= 1] = 1-10e-4
    batch_losses[batch_losses <= 0] = 10e-4

    #B = bmm_model.posterior(batch_losses,1)
    B = bmm_model.look_lookup(batch_losses, bmm_model_maxLoss, bmm_model_minLoss)

    return B


def train_GNNs(model_name, dataset, n_epochs, lr, wd, trained_model_file, mislabel_rate, noise_type, test_target):
    # prepare data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    data, n_classes = get_data(dataset, noise_type, mislabel_rate)
    data.to(device)
    n_features = data.num_features
    print("Data: ", data)

    # prepare model
    if model_name == 'GCN':
        model = GCN(in_channels=n_features, hidden_channels=256, out_channels=n_classes)
    elif model_name == 'GIN':
        model = myGIN(in_channels=n_features, hidden_channels=256, out_channels=n_classes)
    elif model_name == 'GAT':
        model = GAT(in_channels=n_features, hidden_channels=256, out_channels=n_classes)
    elif model_name == 'MLP':
        model = baseMLP([n_features, 256, n_classes])
    elif model_name == 'GraphUNet':
        model = myGraphUNet(in_channels=n_features, hidden_channels=16, out_channels=n_classes)
    model.to(device)
    print("Model: ", model)

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='max')

    # load / train the model
    best_cri = 0
    for epoch in range(n_epochs):
        model.train()
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
            # torch.save(model.state_dict(), trained_model_file)

            all_losses = F.nll_loss(eval_out[data.train_mask], data.y[data.train_mask], reduction='none')
            # all_probs = out
            # arg_entr = torch.max(out[data.train_mask], dim=1)
            # all_argmaxXentropy = F.nll_loss(out[data.train_mask], arg_entr, reduction='none')
            bmm_model, bmm_model_maxLoss, bmm_model_minLoss = track_training_loss(device, all_losses)
            val_losses = F.nll_loss(eval_out[data.val_mask], data.y[data.val_mask], reduction='none')
            B = compute_probabilities_batch(val_losses, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
        # scheduler.step(cri)

    # look up test_target probabilities
    model.eval()
    eval_out = model(data)
    if test_target == 'valid':
        losses = F.nll_loss(eval_out[data.val_mask], data.y[data.val_mask], reduction='none')
    else:
        losses = F.nll_loss(eval_out[data.test_mask], data.y[data.test_mask], reduction='none')
    B = compute_probabilities_batch(losses, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
    return B


if __name__ == "__main__":
    setup_seed(1119)

    parser = argparse.ArgumentParser(description="DY-Bootstrap")
    parser.add_argument("--exp", type=int, default=0)
    parser.add_argument("--dataset", type=str, default='Flickr')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--mislabel_rate", type=float, default=0.1)
    parser.add_argument("--noise_type", type=str, default='symmetric')
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--validation', type=bool, default=True)
    parser.add_argument("--test_target", type=str, default='test')
    args = parser.parse_args()

    ensure_dir('tensorboard_logs')
    log_dir = 'tensorboard_logs/{}-{}-mislabel={}-{}-epochs={}-lr={}-wd={}-exp={}'.format \
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay, args.exp)
    ensure_dir('checkpoints')
    trained_model_file = 'checkpoints/{}-{}-mislabel={}-{}-epochs={}-lr={}-wd={}-exp={}'.format \
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay, args.exp)
    ensure_dir('gnn_results')
    gnn_result_file = 'gnn_results/{}-{}-mislabel={}-{}-epochs={}-lr={}-wd={}-exp={}'.format \
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay, args.exp)
    ensure_dir('mislabel_results')
    # mislabel_result_file = 'mislabel_results/validation-DYB-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format \
    #     (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
    mislabel_result_file = 'mislabel_results/DYB-test={}-{}-{}-mislabel={}-{}-epochs={}-lr={}-wd={}-exp={}'.format \
        (args.test_target, args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay, args.exp)

    B = train_GNNs(args.model, args.dataset, args.n_epochs, args.lr, args.weight_decay, trained_model_file,
                   args.mislabel_rate, args.noise_type, args.test_target)

    print('B: ', B)
    result = B > 0.5
    idx2score = dict()
    for i in range(len(B)):
        if result[i]:
            idx2score[i] = B[i]
    er = [x[0] for x in sorted(idx2score.items(), key=lambda x: x[1], reverse=True)]
    cl_results = pd.DataFrame({'result': pd.Series(result), 'ordered_errors': pd.Series(er), 'score': pd.Series(B)})
    cl_results.to_csv(mislabel_result_file+'.csv', index=False)
    print("DY-bootstrap results saved!")
