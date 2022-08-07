import torch
from torch import nn
from torch.nn import functional as F
from .attention_layer import MultiheadSelfAttention, SelfAttention
from .layers import CNN1d, LSTMModel


FINAL_DIM=512

class CNN1d_Transformer(nn.Module):
    def __init__(self, tstep=4, embed_dim=256, hidden_dim=256):
        super(CNN1d_Transformer,self).__init__()
        self.tstep = tstep
        # Encoder
        self.cnn1d = CNN1d(embed_dim)
        self.lstm = LSTMModel(466, hidden_dim)
        # Decoder
        self.self_attention = MultiheadSelfAttention(hidden_dim*2, hidden_dim*2, heads=4)
        self.norm_layer = nn.LayerNorm([768, hidden_dim*2])
        self.dropout = nn.Dropout(p=0.2)
        self.mlp = SelfAttention(hidden_dim*2, hidden_dim*2)
        # Classifier
        self.fliner = nn.Linear(FINAL_DIM, 4)
        self.sfm = nn.Softmax(dim=1)
        self.logsfm = nn.LogSoftmax(dim=1)
        
    def forward(self, x_):
        """
        This case input shape : (batch, channels=30000, tstep=4)
        """
        # Encoder
        embed = self.cnn_embed(x_)
        # Decoder
        y = self.transfomer(embed) 
        # LSTM Recognizer
        y = y.mean(dim=1)
        y = self.fliner(y)
        return self.sfm(y), self.logsfm(y)

    def transfomer(self, encout, attention_is=False):
        ## attention block
        x = self.norm_layer(encout)
        #if attention_is:
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + encout
        ## mlp block
        y = self.norm_layer(x)
        return y

    def cnn_embed(self, x):
        ss = []
        xs = x.permute(0, 2, 1)
        for i in range(self.tstep):
            xs_ = xs[:, i, :].unsqueeze(1)
            fts = self.cnn1d(xs_)
            ss.append(fts)
        x = torch.cat(ss, dim=1)
        # Standardization
        std, mean = torch.std_mean(x, dim=(1,2), unbiased=False, keepdim=True)
        x = torch.div(x-mean, std)
        x = self.lstm(x)
        return x

