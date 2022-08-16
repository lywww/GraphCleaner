import torch
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv, GATv2Conv
from torch_geometric.nn.models import GIN, MLP, GraphUNet, GAT
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)

    def get_logits(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)


class myGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gin = GIN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=2,
                       out_channels=out_channels, dropout=0.5)  # use the default dropout rate of F.dropout
        # default GIN has relu and dropout (because it uses MLP)
        # while the default GCN, GraphSage, GAT don't have relu and dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gin(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_logits(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gin(x, edge_index)
        return x


class myGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # self.conv1 = GATConv(in_channels, hidden_channels, dropout=0.5)
        # self.conv2 = GATConv(hidden_channels, out_channels, dropout=0.5)
        # self.conv3 = GATConv(hidden_channels, out_channels,dropout=0.5)
        self.gat = GAT(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=3, out_channels=out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv3(x, edge_index)

        x = self.gat(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_logits(self, data):
        x, edge_index = data.x, data.edge_index

        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv3(x, edge_index)

        x = self.gat(x, edge_index)
        return x


class myGraphUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, depth=1):
        super().__init__()
        self.graphunet = GraphUNet(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, depth=depth)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.graphunet(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_logits(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.graphunet(x, edge_index)
        return x


class baseMLP(torch.nn.Module):
    def __init__(self, channel_list, dropout=0.5, relu_first=True):
        super().__init__()
        self.mlp = MLP(channel_list=channel_list, dropout=dropout, relu_first=relu_first, batch_norm=True)

    def forward(self, data):
        y = self.mlp(data.x)
        return F.log_softmax(y, dim=1)

    def get_logits(self, data):
        y = self.mlp(data.x)
        return y
