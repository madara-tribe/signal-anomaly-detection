import torch
from torch import nn
from .mish import Mish

class CNN1d(nn.Module):
    def __init__(self, embed_dim):
        super(CNN1d, self).__init__()
        pool_dim = 4
        self.CE = nn.Sequential(nn.Conv1d(1, embed_dim//4, 16, stride=1, padding=0),
                                    nn.BatchNorm1d(embed_dim//4),
                                    Mish(),
                                    nn.MaxPool1d(pool_dim, stride=pool_dim),
                                    nn.Conv1d(embed_dim//4, embed_dim//2, 8, stride=1, padding=0),
                                    nn.BatchNorm1d(embed_dim//2),
                                    Mish(),
                                    nn.MaxPool1d(pool_dim, stride=pool_dim),
                                    nn.Conv1d(embed_dim//2, embed_dim, 8, stride=1, padding=0),
                                    nn.BatchNorm1d(embed_dim),
                                    Mish(),
                                    nn.MaxPool1d(pool_dim, stride=pool_dim))
        
    def forward(self, x):
        return self.CE(x)

class LSTMModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.gru1 = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
    def forward(self, x):
        gout1, hs1 = self.gru1(x)
        return gout1


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            Mish(),
            nn.Linear(in_dim, out_dim)
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x
