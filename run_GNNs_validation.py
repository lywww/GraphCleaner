import os
import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F

from aum import AUMCalculator
from GNN_models import GCN, myGIN, baseMLP, myGraphUNet, myGAT
from Utils import get_data, to_softmax, create_summary_writer


def train_GNNs(model_name, dataset, noise_type, mislabel_rate, n_epochs, lr, wd, log_dir, trained_model_file,
               test_target, special_set=None):
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
        model = myGAT(in_channels=n_features, hidden_channels=4, out_channels=n_classes)
    elif model_name == 'MLP':
        model = baseMLP([n_features, 256, n_classes])
    elif model_name == 'GraphUNet':
        model = myGraphUNet(in_channels=n_features, hidden_channels=16, out_channels=n_classes)
    model.to(device)
    print("Model: ", model)

    # prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # load / train the model
    isAUM = special_set and 'AUM' in special_set
    if (not special_set) and os.path.exists(trained_model_file):
        print("trained model file: ", trained_model_file)
        model.load_state_dict(torch.load(trained_model_file))
        model.eval()
        best_out = model(data)
    else:
        if isAUM:
            aum_calculator = AUMCalculator('./aum_data', compressed=True)
            sample_ids = np.arange(len(data.y))
            if test_target == 'valid':
                test_ids = (data.val_mask == True).nonzero().reshape(-1).cpu().detach().numpy()
            elif test_target == 'test':
                test_ids = (data.test_mask == True).nonzero().reshape(-1).cpu().detach().numpy()
        best_cri = 0
        best_out = []
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            if isAUM and test_target == 'test':
                loss = F.nll_loss(out[~data.test_mask], data.y[~data.test_mask])
            else:
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            print("Epoch[{}] Loss: {:.2f}".format(epoch + 1, loss))
            writer.add_scalar("training/loss", loss, epoch)
            loss.backward()
            optimizer.step()
            if isAUM:
                aum_input = to_softmax(out.cpu().detach().numpy())
                aum_calculator.update(torch.from_numpy(aum_input).to(device), data.y, sample_ids)
            else:
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
        if isAUM:
            aum_calculator.finalize()

    if isAUM:
        return [], [], test_ids

    # evaluate on validation set
    model.eval()
    predictions = best_out
    y = data.y
    if test_target == 'valid':
        predictions = predictions[data.val_mask]
        y = data.y[data.val_mask]
    elif test_target == 'test':
        predictions = predictions[data.test_mask]
        y = data.y[data.test_mask]

    predictions = to_softmax(predictions.cpu().detach().numpy())
    y = y.cpu().detach().numpy()
    return predictions, y
