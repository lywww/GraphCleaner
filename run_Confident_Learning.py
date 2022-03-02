import argparse
import numpy as np
import pandas as pd

import cleanlab
from cleanlab import baseline_methods
from cleanlab.latent_estimation import compute_confident_joint
from cleanlab.pruning import order_label_errors
from Utils import setup_seed, ensure_dir


def confident_learning(psx, s, mislabel_result_file):
    # Borrowed from https://github.com/cleanlab/cleanlab/tree/master/examples/cifar10
    # cleanlab code for computing the 5 confident learning methods.
    # psx is the n x m matrix of cross-validated predicted probabilities
    # s is the array of noisy labels

    result_file = mislabel_result_file + '.csv'
    # if os.path.exists(result_file):
    #     result = pd.read_csv(result_file)
    #     print("Confident Learning results already existed!")
    #     # return result['baseline_conf_joint_only'], result['baseline_argmax'], result['baseline_cl_pbc'], \
    #     #        result['baseline_cl_pbnr'], result['baseline_cl_both']
    #     return result['result'], result['ordered_errors']

    # Method: C_{\tilde{y}, y^*} (default)
    label_error_mask = np.zeros(len(s), dtype=bool)
    label_error_indices = compute_confident_joint(s, psx, return_indices_of_off_diagonals=True)[1]
    for idx in label_error_indices:
        label_error_mask[idx] = True
    er = order_label_errors(label_error_mask, psx, s)
    # Save the results (True means wrong label)
    cl_results = pd.DataFrame({'result': pd.Series(label_error_mask), 'ordered_errors': pd.Series(er)})
    cl_results.to_csv(result_file, index=False)
    print("Confident Learning results saved!")
    return label_error_mask, er

    # # Method: C_confusion
    # baseline_argmax = baseline_methods.baseline_argmax(psx, s)
    #
    # # Method: CL: PBC
    # baseline_cl_pbc = cleanlab.pruning.get_noise_indices(s, psx, prune_method='prune_by_class')
    #
    # # Method: CL: PBNR
    # baseline_cl_pbnr = cleanlab.pruning.get_noise_indices(s, psx, prune_method='prune_by_noise_rate')
    #
    # # Method: CL: C+NR
    # baseline_cl_both = cleanlab.pruning.get_noise_indices(s, psx, prune_method='both')
    #
    # # Save the results (True means wrong label)
    # cl_results = pd.DataFrame({'baseline_conf_joint_only':baseline_conf_joint_only, 'baseline_argmax':baseline_argmax,
    #                           'baseline_cl_pbc':baseline_cl_pbc, 'baseline_cl_pbnr':baseline_cl_pbnr,
    #                           'baseline_cl_both':baseline_cl_both})
    # cl_results.to_csv(result_file, index=False)
    # print("Confident Learning results saved!")
    # return baseline_conf_joint_only, baseline_argmax, baseline_cl_pbc, baseline_cl_pbnr, baseline_cl_both


if __name__ == "__main__":
    setup_seed(1119)

    parser = argparse.ArgumentParser(description="Confident Learning")
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
    mislabel_result_file = 'mislabel_results/CL-test={}-{}-{}-rate={}-{}-epochs={}-lr={}-wd={}'.format \
        (args.test_target, args.dataset, args.model, args.mislabel_rate, args.noise_type, args.n_epochs, args.lr, args.weight_decay)

    # get the prediction results and save to file
    predictions, noisy_y = train_GNNs(args.model, args.dataset, args.noise_type, args.mislabel_rate, args.n_epochs,
                                      args.lr, args.weight_decay, log_dir, trained_model_file, args.test_target)
    result = pd.DataFrame(data=np.hstack((predictions, noisy_y.reshape((-1,1)))))
    result.to_csv(gnn_result_file+'.csv', index=False, header=None)
    print("{} results saved!".format(args.model))

    # get noise indices
    confident_learning(predictions, noisy_y, mislabel_result_file)
