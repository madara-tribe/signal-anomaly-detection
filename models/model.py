import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary
import os
from .attention import SelfAttention, attention_net
from .parts import CNNEncorder

depths = [64, 128, 256, 512]
start_fm = 64
embed_size = 1024 #64*16

class LSTM_with_atten(nn.Module):
    def __init__(self, config, inc, start_fm, num_cls, embed_size,DO = 0.3):
        super(LSTM_with_atten, self).__init__()
        LSTM_UNITS = start_fm
        self.attention = config.ATTENTION
        self.self_attention = config.SELFATTENTION
        self.cnn = CNNEncorder(inc, start_fm)
        self.selfattension = SelfAttention(LSTM_UNITS, num_cls)
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

        self.linear_pe = nn.Linear(LSTM_UNITS*2, 1)
        self.cls_linear = nn.Linear(LSTM_UNITS*2, num_cls)
        self.final_activate = nn.LogSoftmax(dim=1)
        
    def forward(self, x_, y_):
        x_ = self.cnn(x_)
        y_ = self.cnn(y_)
        x = torch.cat([x_, y_], 3)
        x = torch.reshape(x, (1, embed_size, 1, 1))
        b,f,_,_ = x.shape
        embedding = x.reshape(1,b,f) # torch.Size([1, 1, 512])
        self.lstm1.flatten_parameters()
        h_lstm1, (hidden1, cell1) = self.lstm1(embedding)
        self.lstm2.flatten_parameters()
        h_lstm2, (hidden2, cell2) = self.lstm2(h_lstm1) # torch.Size([1, 1, 128]) torch.Size([2, 1, 64]) torch.Size([2, 1, 64])
        h_conc_linear1  = F.relu(self.linear1(h_lstm1))
        h_conc_linear2  = F.relu(self.linear2(h_lstm2)) # torch.Size([1, 1, 128])
        
        if self.attention:
            _, _, dims = h_lstm1.shape
            h_lstm1 = attention_net(h_lstm1, hidden1.reshape(1, 1, dims)).unsqueeze(0)
            h_lstm2 = attention_net(h_lstm2, hidden2.reshape(1, 1, dims)).unsqueeze(0)
        #print(h_conc_linear.shape, h_lstm.shape)
        hidden = h_conc_linear1 + h_conc_linear2 + h_lstm1 + h_lstm2 #torch.Size([1, 128]) torch.Size([1, 128])
        hidden = hidden.squeeze(0) # torch.Size([1, 128])
        if self.self_attention:
            hidden = self.selfattension(hidden)
        output = self.cls_linear(hidden)
        output = self.final_activate(output) # torch.Size([1, 128])
        return output
 
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



