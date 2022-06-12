import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary
import os
from .attention import SelfAttention, attention_net
from .parts import CNNEncorder

depths = [64, 128, 256, 512]
start_fm = 64
embed_size = 512 #64*8

class LSTM_with_atten(nn.Module):
    def __init__(self, config, inc, start_fm, num_cls, embed_size, DO = 0.3, lstm_use=None):
        super(LSTM_with_atten, self).__init__()
        LSTM_UNITS = 200
        CNN1d_OUT = 8320
        self.lstm_use = lstm_use
        self.attention = config.ATTENTION
        self.self_attention = config.SELFATTENTION
        self.cnn1 = CNNEncorder(inc, start_fm)
        self.cnn2 = CNNEncorder(2, start_fm)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.selfattension = SelfAttention(CNN1d_OUT) #LSTM_UNITS*2)
        # LSTM
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        # 1DCNN
        self.con1d_layer = nn.Sequential(
                    nn.Conv1d(392, 32, kernel_size=2, padding=2),
                    nn.GELU(),
                    nn.AvgPool1d(kernel_size=2),
                    nn.Conv1d(32, 64, kernel_size=2, padding=2),
                    nn.GELU(),
                    nn.AvgPool1d(kernel_size=2))
        # Activation
        self.shape_activate = nn.Sequential(
                                 nn.Linear(embed_size, 2),
                                 nn.Sigmoid()) #nn.LogSoftmax(dim=1)
        self.color_activate = nn.Sequential(
                                 nn.Linear(CNN1d_OUT, num_cls),
                                 nn.Sigmoid())
        
    def forward(self, x_, y_):
        x_ = self.cnn1(x_)
        y_ = self.cnn2(y_)
        encoder_output = torch.cat([x_, y_], 3)
        x = self.avgpool(encoder_output) # torch.Size([1, 512, 1, 1])
        # torch.Size([1, 512, 7, 7])
        with torch.no_grad():
            b,f,_,_ = x.shape
            embedding = x.reshape(1,b,f) # torch.Size([1, 1, 512])
        
        if self.lstm_use:
            self.lstm1.flatten_parameters()
            h_lstm1, (hidden1, cell1) = self.lstm1(embedding)
            #_, _, dims = h_lstm1.shape
            #h_lstm1 = attention_net(h_lstm1, hidden1.reshape(1, 1, dims)).unsqueeze(0)
            self.lstm2.flatten_parameters()
            h_lstm2, (hidden2, cell2) = self.lstm2(h_lstm1) # torch.Size([1, 1, 128]) torch.Size([2, 1, 64]) torch.Size([2, 1, 64])
            h_conc_linear1  = F.relu(self.linear1(h_lstm1))
            h_conc_linear2  = F.relu(self.linear2(h_lstm2))
            hidden = h_conc_linear1 + h_conc_linear2 + h_lstm1 + h_lstm2 #torch.Size([1, 1, 128])
        else:
            b, e, f, g = encoder_output.shape
            cnn_emb = encoder_output.reshape(b, f*g, e)
            hidden = self.con1d_layer(cnn_emb)
            hidden = hidden.reshape(b, 8320).unsqueeze(0)
        
        if self.self_attention:
            new_hidden = self.selfattension(hidden)
        # color
        color_label = self.color_activate(new_hidden) # torch.Size([1, 128])
        # shape 
        shape_label = self.shape_activate(embedding.squeeze(0))
        return color_label, shape_label
     
def main(model, input_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H = W = 256
    in_size = 1
    CLS=128
    input_size = (1, in_size, H, W)
    inp1 = torch.rand(input_size, dtype=torch.float32).to(device)
    inp2 = torch.rand(input_size, dtype=torch.float32).to(device)
    model = LSTM_with_atten(inc=in_size, start_fm = start_fm, num_cls=CLS, embed_size= embed_size).to(device)
    out = model(inp1, inp2)
