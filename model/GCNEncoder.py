import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.nn.inits import reset, uniform
import random
import numpy as np

class POIEncoder(nn.Module):
    """POI GCN encoder"""
    def __init__(self, in_channels, hidden_channels):
        super(POIEncoder, self).__init__()
        self.conv = GATConv(in_channels, hidden_channels, heads=4, cached=False, bias=True)
        self.norm = nn.LayerNorm(hidden_channels*4)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.norm(x)

        return x

class GCNClassification(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCNClassification, self).__init__()
        self.hidden_channels = hidden_channels
        self.poi_encoder = POIEncoder(7, hidden_channels)
        self.linear = nn.Linear(hidden_channels*4, 7)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.poi_encoder)

    def forward(self, data):
        poi_emb = self.poi_encoder(data.x, data.edge_index, data.edge_weight)
        poi_emb = self.linear(poi_emb)
        labels = nn.Softmax(dim=1)(poi_emb)
        labels = labels[data.ids]
        return labels

    def loss(self, poi_emb, y_labels):
        """compute the cross-entropy loss"""
        from sklearn.utils import class_weight

        y = y_labels.cpu().numpy()
        class_weights=class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        class_weights=torch.tensor(class_weights,dtype=torch.float)
        
        loss = nn.CrossEntropyLoss(weight=class_weights)
        return loss(poi_emb, y_labels)
        
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)