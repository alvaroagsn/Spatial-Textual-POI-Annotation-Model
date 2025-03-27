import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.inits import reset


class POIEncoder(nn.Module):
    """POI GCN encoder - Modelo base permanece igual"""
    def __init__(self, in_channels, hidden_channels):
        super(POIEncoder, self).__init__()
        self.conv = GATConv(in_channels, hidden_channels, heads=1, bias=True)  
        self.norm = nn.LayerNorm(hidden_channels * 1)  

    def forward(self, x, edge_index):
        """
        Executa a convolução GAT e normaliza a saída
        Removemos edge_weight pois DGI não usa pesos explícitos de arestas.
        """
        print(f"Input x shape: {x.shape if x is not None else 'None'}")
        print(f"Input edge_index shape: {edge_index.shape if edge_index is not None else 'None'}")

        x = self.conv(x, edge_index)

        print(f"Output x shape after GATConv: {x.shape if x is not None else 'None'}")

        x = self.norm(x)

        print(f"Output x shape after LayerNorm: {x.shape if x is not None else 'None'}")

        return x


class Readout(nn.Module):
    """Função de leitura global (Readout)"""
    def forward(self, x):
        print(f"Input x shape: {x.shape if x is not None else 'None'}")
        readout_result = torch.sigmoid(torch.mean(x, dim=0))
        print(f"Output shape: {readout_result.shape if readout_result is not None else 'None'}")
        return readout_result


class Discriminator(nn.Module):
    """Discriminador para aprendizado contrastivo"""
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(hidden_dim * 1, 1)  # O GAT usa 4 cabeças, então o embedding é 4x hidden_dim

    def forward(self, summary, x):

        summary = summary.expand_as(x)
        score = self.linear(summary * x) 
        print(f"Score shape: {score.shape if score is not None else 'None'}")

        return score

class DGIModule(nn.Module):
    """Deep Graph Infomax (DGI) - Modelo final"""
    def __init__(self, hidden_channels):
        super(DGIModule, self).__init__()
        self.hidden_channels = hidden_channels
        
        self.poi_encoder = POIEncoder(7, hidden_channels)

        self.readout = Readout()

        self.discriminator = Discriminator(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):

        reset(self.poi_encoder)
        nn.init.xavier_uniform_(self.discriminator.linear.weight)  # Xavier ajuda a evitar explosão de gradientes

    def forward(self, data):
        """Gera embeddings para o grafo"""
        print(f"Data.x shape: {data.x.shape if data.x is not None else 'None'}")
        print(f"Data.edge_index shape: {data.edge_index.shape if data.edge_index is not None else 'None'}")

        x = self.poi_encoder(data.x, data.edge_index)  # Passa os nós pelo encoder
        summary = self.readout(x)  # Obtém a representação global do grafo

        permuted_idx = torch.randperm(x.size(0))  # Gera um índice aleatório para embaralhar
        x_corrupted = x[permuted_idx]  # Embeddings corrompidos

        pos_score = self.discriminator(summary, x)  # Score para embeddings reai
        neg_score = self.discriminator(summary, x_corrupted)  # Score para embeddings corrompidos

        return pos_score, neg_score

    def loss(self, pos_score, neg_score):
        """
        Função de perda baseada em Binary Cross Entropy (BCE)
        """
        pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))  # BCE para positivos
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))  # BCE para negativos

        return pos_loss + neg_loss  # Soma das perdas

    def __repr__(self):
        return f'{self.__class__.__name__}({self.hidden_channels})'
