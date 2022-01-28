import os

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Flickr, PPI

# dataset = PygNodePropPredDataset(name = "ogbn-arxiv")
# split_idx = dataset.get_idx_split()
# train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
# graph = dataset[0]  # pyg graph object

path = os.path.join('./dataset', 'PPI')
dataset = PPI(path)
