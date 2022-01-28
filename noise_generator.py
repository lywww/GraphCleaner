import os
import argparse
import random
import json
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset


dataset2classes = {'Flickr': 7, 'Reddit2': 41, 'Cora': 7, 'CiteSeer': 6, 'PubMed': 3}


def generate_noise(args):
    print("Loading {}...".format(args.dataset))
    if args.dataset == 'ogbn-papers100M':  # haven't completed, require modification when used
        dataset = PygNodePropPredDataset(name=args.dataset, root=args.data_dir)
        n_classes = dataset.num_classes
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"].numpy().tolist(), split_idx["valid"].numpy().tolist(), \
                                         split_idx["test"].numpy().tolist()
        print("Loaded! There are {} training data, {} validation data, {} test data.".format
              (len(train_idx), len(valid_idx), len(test_idx)))

        print("Choosing samples with mislabelling rate {}...".format(args.mislabel_rate))
        train_sample_idx = random.sample(train_idx, np.round(args.mislabel_rate * len(train_idx)))
        valid_sample_idx = random.sample(valid_idx, np.round(args.mislabel_rate * len(valid_idx)))

        print("Changing the label of samples symmetrically...")
        label_dir = os.path.join(args.data_dir, args.dataset, 'raw')
        label_file = np.load(os.path.join(label_dir, 'node-label.npz'))  # need modification if ogbn-arxiv
        for idx in train_sample_idx + valid_sample_idx:
            new_label = random.randint(0, n_classes-1)  # label: 0-171
            label = label_file['node_label'][idx]
            if new_label < label:
                label_file['node_label'][idx] = new_label
            else:
                label_file['node_label'][idx] = new_label + 1
        np.save(os.path.join(label_dir, 'corrupted-node-label.npz'), label_file)

    elif args.dataset == 'Flickr' or args.dataset == 'Reddit2':
        n_classes = dataset2classes[args.dataset]
        dir = os.path.join('./dataset', args.dataset, 'raw')
        with open(os.path.join(dir, 'class_map.json'), 'r') as f:
            class_map = json.load(f)
        with open(os.path.join(dir, 'role.json'), 'r') as f:
            role = json.load(f)
        print("Loaded! There are {} training data, {} validation data, {} test data.".format
              (len(role['tr']), len(role['va']), len(role['te'])))

        print("Choosing samples with mislabelling rate {}...".format(args.mislabel_rate))
        tr_label_data = dict(zip(range(n_classes), [[] for _ in range(n_classes)]))
        va_label_data = dict(zip(range(n_classes), [[] for _ in range(n_classes)]))
        te_label_data = dict(zip(range(n_classes), [[] for _ in range(n_classes)]))
        for data in role['tr']:
            tr_label_data[class_map[str(data)]].append(data)
        for data in role['va']:
            va_label_data[class_map[str(data)]].append(data)
        for data in role['te']:
            te_label_data[class_map[str(data)]].append(data)
        samples = []
        for k, v in tr_label_data.items():
            samples += list(np.random.choice(v, size=int(len(v)*args.mislabel_rate), replace=False))
        for k, v in va_label_data.items():
            samples += list(np.random.choice(v, size=int(len(v)*args.mislabel_rate), replace=False))
        for k, v in te_label_data.items():
            samples += list(np.random.choice(v, size=int(len(v)*args.mislabel_rate), replace=False))

        noisy_class_map = copy.deepcopy(class_map)
        if args.noise_type == 'symmetric':
            print("Changing the label of samples symmetrically...")
            for sample in samples:
                ori_label = class_map[str(sample)]
                labels = [i for i in range(n_classes)]
                labels.remove(ori_label)
                noisy_label = np.random.choice(a=labels, size=1, replace=False)
                noisy_class_map[str(sample)] = int(noisy_label[0])
        else:
            print("Changing the label of samples asymmetrically...")
            for sample in samples:
                ori_label = class_map[str(sample)]
                if ori_label == n_classes-1:
                    noisy_label = 0
                else:
                    noisy_label = ori_label + 1
                noisy_class_map[str(sample)] = noisy_label

        print("Saving the noisy label...")
        with open(os.path.join(dir, 'noisy_class_map_'+args.noise_type+'_'+str(args.mislabel_rate)+'.json'), 'w') as f:
            json.dump(noisy_class_map, f)

        print("Drawing the noise transition matrix...")
        real_transit = np.zeros((n_classes, n_classes))
        for k in role['tr']:
            real_transit[class_map[str(k)]][noisy_class_map[str(k)]] += 1
        real_transit = real_transit / np.sum(real_transit, axis=1).reshape(-1, 1)
        plt.figure()
        sns.heatmap(real_transit, cmap='PuBu', vmin=0, vmax=1, linewidth=1, annot=True)
        plt.title('Groundtruth Noise Transition Matrix')
        plt.savefig('GT_Noise_Matrix_' + args.dataset + '_' + args.noise_type + '_' + str(args.mislabel_rate) + '.jpg',
                    bbox_inches='tight')
        plt.show()

    elif args.dataset == 'Cora' or args.dataset == 'CiteSeer' or args.dataset == 'PubMed':
        dataset = Planetoid(root='./dataset', name=args.dataset)
        data = dataset[0]
        n_classes = dataset2classes[args.dataset]
        train_mask = np.ones(len(data.train_mask), dtype=bool)
        train_mask[data.val_mask] = False
        train_mask[data.test_mask] = False
        print("Loaded! There are {} training data, {} validation data, {} test data.".format
              (sum(train_mask), sum(data.val_mask), sum(data.test_mask)))

        print("Choosing samples with mislabelling rate {}...".format(args.mislabel_rate))
        tr_label_data = dict(zip(range(n_classes), [[] for _ in range(n_classes)]))
        va_label_data = dict(zip(range(n_classes), [[] for _ in range(n_classes)]))
        te_label_data = dict(zip(range(n_classes), [[] for _ in range(n_classes)]))
        for i, y in enumerate(data.y):
            if train_mask[i]:
                tr_label_data[y.item()].append(i)
            elif data.val_mask[i]:
                va_label_data[y.item()].append(i)
            elif data.test_mask[i]:
                te_label_data[y.item()].append(i)
        samples = []
        for k, v in tr_label_data.items():
            samples += list(np.random.choice(v, size=int(len(v) * args.mislabel_rate), replace=False))
        for k, v in va_label_data.items():
            samples += list(np.random.choice(v, size=int(len(v) * args.mislabel_rate), replace=False))
        for k, v in te_label_data.items():
            samples += list(np.random.choice(v, size=int(len(v) * args.mislabel_rate), replace=False))
        print("num of samples: ", len(samples), len(set(samples)))

        noisy_class_map = dict(zip(range(len(data.y)), [y.item() for y in data.y]))
        if args.noise_type == 'symmetric':
            print("Changing the label of samples symmetrically...")
            for sample in samples:
                ori_label = noisy_class_map[sample]
                labels = [i for i in range(n_classes)]
                labels.remove(ori_label)
                noisy_label = np.random.choice(a=labels, size=1, replace=False)
                noisy_class_map[sample] = int(noisy_label[0])
        else:
            print("Changing the label of samples asymmetrically...")
            for sample in samples:
                ori_label = noisy_class_map[sample]
                if ori_label == n_classes - 1:
                    noisy_label = 0
                else:
                    noisy_label = ori_label + 1
                noisy_class_map[sample] = noisy_label

        print("Saving the noisy label...")
        with open(os.path.join('./dataset', args.dataset, 'raw/noisy_class_map_' + args.noise_type + '_' +
                              str(args.mislabel_rate) + '.json'), 'w') as f:
            json.dump(noisy_class_map, f)

        print("Drawing the noise transition matrix...")
        real_transit = np.zeros((n_classes, n_classes))
        for k, v in enumerate(data.y):
            if train_mask[k]:
                real_transit[v][noisy_class_map[k]] += 1
        real_transit = real_transit / np.sum(real_transit, axis=1).reshape(-1, 1)
        plt.figure()
        sns.heatmap(real_transit, cmap='PuBu', vmin=0, vmax=1, linewidth=1, annot=True)
        plt.title('Groundtruth Noise Transition Matrix')
        matrix_name = 'GT_Noise_Matrix_' + args.dataset + '_' + args.noise_type + '_' + str(args.mislabel_rate)
        plt.savefig(matrix_name + '.jpg', bbox_inches='tight')
        plt.show()
        np.save(matrix_name + '.npy', real_transit.T)


if __name__ == "__main__":
    random.seed(1119)
    np.random.seed(1119)

    parser = argparse.ArgumentParser(description="Generate Noises")
    parser.add_argument("--dataset", type=str, default='Flickr')
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--mislabel_rate", type=float, default=0.1)
    parser.add_argument("--noise_type", type=str, default='symmetric')
    args = parser.parse_args()

    generate_noise(args)
