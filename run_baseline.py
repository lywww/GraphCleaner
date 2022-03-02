import argparse
import numpy as np
import pandas as pd

from Utils import setup_seed, ensure_dir


def baseline(predictions, noisy_y, mislabel_result_file):
    result_file = mislabel_result_file + '.csv'
    # if os.path.exists(result_file):
    #     result = pd.read_csv(result_file)
    #     print("Baseline results already existed!")
    #     return result['result'], result['ordered_errors']

    result = []
    idx2score = dict()
    for i in range(len(noisy_y)):
        if min(predictions[i]) == predictions[i][noisy_y[i]]:
            result.append(True)
            idx2score[i] = predictions[i][noisy_y[i]]
        else:
            result.append(False)
    er = [x[0] for x in sorted(idx2score.items(), key=lambda x:x[1])]

    # Save the results (True means wrong label)
    cl_results = pd.DataFrame({'result': pd.Series(result), 'ordered_errors': pd.Series(er)})
    cl_results.to_csv(result_file, index=False)
    print("Baseline results saved!")
    return result, er


if __name__ == "__main__":
    setup_seed(1119)

    parser = argparse.ArgumentParser(description="Baseline")
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
    log_dir = 'tensorboard_logs/{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format\
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
    ensure_dir('checkpoints')
    trained_model_file = 'checkpoints/{}-{}-rate={}-{}-epochs={}-bs=2048-lr={}-wd={}'.format\
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
    ensure_dir('gnn_results')
    gnn_result_file = 'gnn_results/{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format\
        (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
    ensure_dir('mislabel_results')

    from run_GNNs_validation import train_GNNs
    mislabel_result_file = 'mislabel_results/baseline-test={}-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format \
        (args.test_target, args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)

    # get the prediction results and save to file
    predictions, noisy_y = train_GNNs(args.model, args.dataset, args.noise_type, args.mislabel_rate, args.n_epochs,
                                      args.lr, args.weight_decay, log_dir, trained_model_file, args.test_target)
    result = pd.DataFrame(data=np.hstack((predictions, noisy_y.reshape((-1,1)))))
    result.to_csv(gnn_result_file+'.csv', index=False, header=None)
    print("{} results saved!".format(args.model))

    # get noise indices
    baseline(predictions, noisy_y, mislabel_result_file)
