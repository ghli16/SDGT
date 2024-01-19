from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random
import torch_geometric.nn as pyg_nn
from utils import args, set_seed
class GCN(torch.nn.Module):
    def __init__(self, node_features, gcn1_features, gcn2_features, gcn3_features):
        super().__init__()
        set_seed()
        self.drop1 = nn.Dropout(p=0.2)
        self.activate = F.elu

        self.GCN1 = GCNConv(node_features, gcn1_features)
        self.GCN2 = GCNConv(gcn1_features, gcn2_features)
        self.GCN3 = GCNConv(gcn2_features, gcn3_features)


    def forward(self, data, x):
        edge_index, batch = data.edge_index, data.batch

        x = self.activate(self.GCN1(x, edge_index))
        x = self.drop1(x)
        x = F.dropout(x, training=self.training)
        x = self.activate(self.GCN2(x, edge_index))
        # x = self.activate(self.GCN3(x, edge_index))
        return x