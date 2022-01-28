import os
import json
import argparse
import random
import numpy as np
import pandas as pd

import torch
from NeighborAgg.NeighborAgg.methods.trustscore import TrustScore_learnable
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


if __name__ == "__main__":
    setup_seed(1119)

    parser = argparse.ArgumentParser(description="NEIGHBORAGG-CMD")
    parser.add_argument("--dataset", type=str, default='Flickr')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--mislabel_rate", type=float, default=0.1)
    parser.add_argument("--noise_type", type=str, default='symmetric')
    parser.add_argument("--model", type=str, default='GCN')
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--special_set", type=str, default='nbagg')
    parser.add_argument('--validation', type=bool, default=False)
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
    if args.validation:
        from run_GNNs_validation import train_GNNs, get_data
        mislabel_result_file = 'mislabel_results/validation-nbaggcmd-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format \
            (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
        # get the prediction results and save to file
        predictions, noisy_y, tr_predictions, tr_noisy_y = train_GNNs(args.model, args.dataset, args.noise_type,
                                                                      args.mislabel_rate, args.n_epochs, args.lr,
                                                                      args.weight_decay, log_dir, trained_model_file,
                                                                      args.special_set)
    else:
        from run_GNNs import train_GNNs, get_data
        mislabel_result_file = 'mislabel_results/nbaggcmd-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format \
            (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
        # get the prediction results and save to file
        predictions, noisy_y = train_GNNs(args.model, args.dataset, args.noise_type, args.mislabel_rate, args.n_epochs,
                                         args.lr, args.weight_decay, log_dir, trained_model_file)


    result = pd.DataFrame(data=np.hstack((predictions, noisy_y.reshape((-1,1)))))
    result.to_csv(gnn_result_file+'.csv', index=False, header=None)
    print("{} results saved!".format(args.model))

    # get noise indices
    result_file = mislabel_result_file + '.json'
    # if os.path.exists(result_file):
    #     print("NeighborAgg-CMD results already existed!")
    #     with open(result_file) as f:
    #         result = json.load(f)
    #         learnable_score = np.array(result[0]['learnable_score'])
    #         mispredicted_idx = np.array(result[0]['mispredicted_idx'])
    # else:
    ts_args = {
        "trust_model": "LR",
        "kdtree_by_class": True,
        "graph_model": "GMMConvNet",
        "num_epochs": 1000,
        "filtering": "density",
        "TS_alpha": 0.0625,
        'similarity_T': 0,
        'val_k': 5,
        "writer": None
    }
    learnable_ts = TrustScore_learnable(ts_args)
    print('building kdtree')
    data, n_classes = get_data(args.dataset, args.noise_type, args.mislabel_rate)
    learnable_ts.fit(data.x[data.train_mask], data.y[data.train_mask], n_classes)
    print('fitting')
    if args.validation:
        learnable_ts.fit_learnable(data.x[data.train_mask], data.y[data.train_mask], tr_predictions)
    else:
        learnable_ts.fit_learnable(data.x[data.train_mask], data.y[data.train_mask], predictions)
    print('giving score')
    if args.validation:
        learnable_score = learnable_ts.get_score(data.x[data.val_mask], data.y[data.val_mask], predictions)
    else:
        learnable_score = learnable_ts.get_score(data.x[data.train_mask], data.y[data.train_mask], predictions)
    mispredicted_idx = np.argwhere((np.argmax(predictions, axis=1) == noisy_y) == True).reshape(-1)
    result_dict = dict()
    result_dict['learnable_score'] = learnable_score.tolist()
    result_dict['mispredicted_idx'] = mispredicted_idx.tolist()
    result = [result_dict]
    with open(result_file, 'w') as f:
        json.dump(result, f)
    print("NeighborAgg-CMD results saved!")

    idx2score = dict(zip(mispredicted_idx, learnable_score[mispredicted_idx]))
    ordered_idx = [x[0] for x in sorted(idx2score.items(), key=lambda x: x[1], reverse=True)]
    cal_patk(ordered_idx, get_ytest(args.dataset, args.noise_type, args.mislabel_rate, args.validation))
