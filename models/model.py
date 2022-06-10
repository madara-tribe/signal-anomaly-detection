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
    def __init__(self, config, inc, start_fm, num_cls, embed_size, DO = 0.3):
        super(LSTM_with_atten, self).__init__()
        LSTM_UNITS = 200
        self.attention = config.ATTENTION
        self.self_attention = config.SELFATTENTION
        self.cnn1 = CNNEncorder(inc, start_fm)
        self.cnn2 = CNNEncorder(2, start_fm)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.selfattension = SelfAttention(LSTM_UNITS, num_cls)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

        self.shape_activate = nn.Sequential(
                                 nn.Linear(embed_size, 2),
                                 nn.Sigmoid()) #nn.LogSoftmax(dim=1)
        self.final_activate = nn.Sequential(
                                 nn.Linear(LSTM_UNITS*2, num_cls),
                                 nn.Sigmoid())
        
    def forward(self, x_, y_):
        x_ = self.cnn1(x_)
        y_ = self.cnn2(y_)
        # torch.Size([1, 512, 7, 7])
        x = torch.cat([x_, y_], 3)
        x = self.avgpool(x) # torch.Size([1, 512, 1, 1])
        b,f,_,_ = x.shape
        embedding = x.reshape(1,b,f) # torch.Size([1, 1, 512])
        self.lstm1.flatten_parameters()
        h_lstm1, (hidden1, cell1) = self.lstm1(embedding)
        #if self.attention:
         #   _, _, dims = h_lstm1.shape
          #  h_lstm1 = attention_net(h_lstm1, hidden1.reshape(1, 1, dims)).unsqueeze(0)
        self.lstm2.flatten_parameters()
        h_lstm2, (hidden2, cell2) = self.lstm2(h_lstm1) # torch.Size([1, 1, 128]) torch.Size([2, 1, 64]) torch.Size([2, 1, 64])
        #if self.attention:
         #   _, _, dims = h_lstm2.shape
          #  h_lstm2 = attention_net(h_lstm2, hidden2.reshape(1, 1, dims)).unsqueeze(0)
        hidden =  h_lstm1 + h_lstm2 #torch.Size([1, 128]) torch.Size([1, 128])
        if self.self_attention:
            hidden = self.selfattension(hidden)
        # color      
        color_label = self.final_activate(hidden) # torch.Size([1, 128])
        # shape 
        shape_label = self.shape_activate(embedding.squeeze(0))
        return color_label, shape_label
 
def main(model, input_size):
    inp1 = torch.rand(input_size, dtype=torch.float32).to(device)
    inp2 = torch.rand(input_size, dtype=torch.float32).to(device)
    out = model(inp1, inp2)
    print(out.shape)
    
if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H = W = 256
    in_size = 1
    CLS=128
    input_size = (1, in_size, H, W)
    #cnn = CNNEncorder(inc=in_size, start_fm = start_fm).to(device)
    model = LSTM_with_atten(inc=in_size, start_fm = start_fm, num_cls=CLS, embed_size= embed_size).to(device)

