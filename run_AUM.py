import argparse
import copy
import numpy as np
import pandas as pd

from Utils import get_data, setup_seed, ensure_dir


def get_threshold_samples(dataset, test_target):
    ensure_dir('aum_data')
    file_name = './aum_data/' + dataset + '_aum_threshold_samples.csv'
    # if os.path.exists(file_name):
    #     print("Threshold samples info already generated!")
    #     return

    data, n_classes = get_data(dataset)
    if test_target == 'valid':
        train_idx = np.argwhere(data.train_mask.numpy() == True).reshape(-1)
    elif test_target == 'test':
        train_idx = np.argwhere(data.test_mask.numpy() == False).reshape(-1)
    first_threshold_samples = np.random.choice(a=train_idx, size=len(train_idx)//(n_classes+1), replace=False)
    second_threshold_samples = np.random.choice(a=np.setdiff1d(train_idx, first_threshold_samples),
                                                size=len(train_idx)//(n_classes+1), replace=False)
    threshold_samples = pd.DataFrame({'first_threshold_samples': first_threshold_samples,
                                        'second_threshold_samples': second_threshold_samples})
    threshold_samples.to_csv(file_name, index=False)
    print("Threshold samples info saved!")


def aum(dataset, special_set, mislabel_result_file):
    # Calculate the 99-percentile threshold
    threshold_samples_file = './aum_data/' + dataset + '_aum_threshold_samples.csv'
    threshold_samples = pd.read_csv(threshold_samples_file)
    aum_values = pd.read_csv('./aum_data/aum_values.csv')
    idx2aum = dict(zip(aum_values['sample_id'], aum_values['aum']))
    threshold_aums = []
    if special_set[-1] == '1':
        threshold_samples = threshold_samples['first_threshold_samples'].values
        for idx in threshold_samples:
            threshold_aums.append(idx2aum[idx])
    else:
        threshold_samples = threshold_samples['second_threshold_samples'].values
        for idx in threshold_samples:
            threshold_aums.append(idx2aum[idx])
    threshold_aums.sort()
    threshold = threshold_aums[int(len(threshold_aums)*0.99)]
    print('threshold is {}'.format(threshold))

    # Filter out the mislabelled samples
    idx2result = copy.deepcopy(idx2aum)
    for k,v in idx2aum.items():
        if k in threshold_samples:
            continue
        if v <= threshold:
            idx2result[k] = True
        else:
            idx2result[k] = False
    result = [pair[1] for pair in sorted(idx2result.items(), key=lambda x: x[0])]
    aum = [idx2aum[i] for i in sorted(idx2aum)]

    # Save the results (True means wrong label)
    result_file = mislabel_result_file + '.csv'
    if special_set[-1] == '2':
        previous_result = pd.read_csv(result_file)
        for i in range(len(result)):
            if result[i] is not True and result[i] is not False:
                result[i] = previous_result['result'][i]
                aum[i] = previous_result['aum'][i]

    idx2score = dict()
    for i in range(len(result)):
        if result[i]:
            idx2score[i] = aum[i]
    er = [x[0] for x in sorted(idx2score.items(), key=lambda x: x[1])]
    cl_results = pd.DataFrame({'result': pd.Series(result), 'aum': pd.Series(aum), 'ordered_errors': pd.Series(er)})
    cl_results.to_csv(result_file, index=False)
    print("{} results saved!".format(special_set))
    return result, er


def validation_aum(dataset, special_set, test_idx, mislabel_result_file):
    # Calculate the 99-percentile threshold from the training set
    threshold_samples_file = './aum_data/' + dataset + '_aum_threshold_samples.csv'
    threshold_samples = pd.read_csv(threshold_samples_file)
    aum_values = pd.read_csv('./aum_data/aum_values.csv')
    idx2aum = dict(zip(aum_values['sample_id'], aum_values['aum']))
    threshold_aums = []
    threshold_samples = threshold_samples['first_threshold_samples'].values
    for idx in threshold_samples:
        threshold_aums.append(idx2aum[idx])
    threshold_aums.sort()
    threshold = threshold_aums[int(len(threshold_aums)*0.99)]
    print('threshold is {}'.format(threshold))

    # Filter out the mislabelled samples in the target test set
    result = []
    aum = []
    idx2score = dict()
    for i, v_idx in enumerate(test_idx):
        mislabel = idx2aum[v_idx] <= threshold
        result.append(mislabel)  # ordered by val idx
        aum.append(idx2aum[v_idx])  # ordered by val idx
        if mislabel:
            idx2score[i] = idx2aum[v_idx]  # mislabel idx: aum_score

    # Save the results (True means wrong label)
    result_file = mislabel_result_file + '.csv'
    er = [x[0] for x in sorted(idx2score.items(), key=lambda x: x[1])]
    cl_results = pd.DataFrame({'result': pd.Series(result), 'score': pd.Series(aum), 'ordered_errors': pd.Series(er)})
    cl_results.to_csv(result_file, index=False)
    print("{} results saved!".format(special_set))
    return result, er


if __name__ == "__main__":
    setup_seed(1119)

    parser = argparse.ArgumentParser(description="AUM")
    parser.add_argument("--dataset", type=str, default='Flickr')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--special_set", type=str, default='AUM1')
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
    log_dir = 'tensorboard_logs/{}-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format\
        (args.special_set, args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
    ensure_dir('checkpoints')
    trained_model_file = 'checkpoints/{}-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format\
        (args.special_set, args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)

    # generate threshold samples
    get_threshold_samples(args.dataset, args.test_target)

    ensure_dir('mislabel_results')
    # if args.validation:
    #     from run_GNNs_validation import train_GNNs
    #     mislabel_result_file = 'mislabel_results/validation-AUM-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format \
    #         (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
    #     # get the prediction results and save to file
    #     predictions, noisy_y, val_idx = train_GNNs(args.model, args.dataset, args.noise_type, args.mislabel_rate,
    #                                                args.n_epochs, args.lr, args.weight_decay, log_dir,
    #                                                trained_model_file, args.special_set)
    #     # get noise indices
    #     validation_aum(args.dataset, args.special_set, val_idx, mislabel_result_file)
    # else:
    #     from run_GNNs import train_GNNs
    #     mislabel_result_file = 'mislabel_results/AUM-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format \
    #         (args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
    #     # get the prediction results and save to file
    #     predictions, noisy_y = train_GNNs(args.model, args.dataset, args.noise_type, args.mislabel_rate, args.n_epochs,
    #                                       args.lr, args.weight_decay, log_dir, trained_model_file, args.special_set)
    #     # get noise indices
    #     aum(args.dataset, args.special_set, mislabel_result_file)
    from run_GNNs_validation import train_GNNs
    mislabel_result_file = 'mislabel_results/AUM-test={}-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format\
        (args.test_target, args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)
    # get the prediction results and save to file
    predictions, noisy_y, test_idx = train_GNNs(args.model, args.dataset, args.noise_type, args.mislabel_rate,
                                               args.n_epochs, args.lr, args.weight_decay, log_dir,
                                               trained_model_file, args.test_target, args.special_set)
    # get noise indices
    validation_aum(args.dataset, args.special_set, test_idx, mislabel_result_file)
