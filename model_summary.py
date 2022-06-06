import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary
import os


depths = [64, 128, 256, 512]
start_fm = 64
embed_size = 512 #64*8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H = W = 256
in_size = 3
CLS=128

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU())
        
    def forward(self, x):
        x = self.conv(x)
        return x

def attention_net(lstm_output, final_state):
    lstm_output = lstm_output.permute(1, 0, 2)
    hidden = final_state.squeeze(0)
    attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
    soft_attn_weights = F.softmax(attn_weights, dim=1)
    new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                 soft_attn_weights.unsqueeze(2)).squeeze(2)
    return new_hidden_state



def attention(lstm_output, final_state):
    lstm_output = lstm_output.permute(1, 0, 2)
    merged_state = torch.cat([s for s in final_state], 1)
    merged_state = merged_state.squeeze(0).unsqueeze(2)
    weights = torch.bmm(lstm_output, merged_state)
    weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
    return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)



class CNN_LSTM_with_atten(nn.Module):
    def __init__(self, inc, num_cls, embed_size=embed_size, LSTM_UNITS=64, DO = 0.3):
        super(CNN_LSTM_with_atten, self).__init__()
        self.double_conv1 = double_conv(inc, start_fm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.double_conv2 = double_conv(start_fm, start_fm * 2)
        #Max Pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        #Convolution 3
        self.double_conv3 = double_conv(start_fm * 2, start_fm * 4)
        #Max Pooling 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        #Convolution 4
        self.double_conv4 = double_conv(start_fm * 4, start_fm * 8)
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

        self.linear_pe = nn.Linear(LSTM_UNITS*2, 1)
        self.linear_global = nn.Linear(LSTM_UNITS*2, num_cls)
        self.final_activate = nn.Softmax()
        
    def forward(self, x, lengths=None):
        x = self.double_conv1(x)
        x = self.maxpool1(x)

        x = self.double_conv2(x)
        x = self.maxpool2(x)

        x = self.double_conv3(x)
        x = self.maxpool3(x)

        x = self.double_conv4(x)
        embedding = self.maxpool4(x) # torch.Size([1, 512, 14, 14])
        #print(embedding.shape)
        embedding = self.avgpool(embedding) # torch.Size([1, 512, 1, 1])
        #print(embedding.shape)
        b,f,_,_ = embedding.shape
        embedding = embedding.reshape(1,b,f) # torch.Size([1, 1, 512])
        #print(embedding.shape)
        self.lstm1.flatten_parameters()
        h_lstm1, (hidden1, cell1) = self.lstm1(embedding)
        self.lstm2.flatten_parameters()
        h_lstm2, (hidden2, cell2) = self.lstm2(h_lstm1) # torch.Size([1, 1, 128]) torch.Size([2, 1, 64]) torch.Size([2, 1, 64])
        #print(h_lstm1.shape, hidden1.shape, cell1.shape)
        h_conc_linear1  = F.relu(self.linear1(h_lstm1))
        h_conc_linear2  = F.relu(self.linear2(h_lstm2)) # torch.Size([1, 1, 128])
        
        h_conc_linear = attention_net(h_conc_linear1, h_conc_linear2)
        h_lstm = attention_net(h_lstm1, h_lstm2)
        #print(h_conc_linear.shape, h_lstm.shape)
        hidden = h_conc_linear + h_lstm #torch.Size([1, 128]) torch.Size([1, 128])
        hidden = hidden.unsqueeze(0) # torch.Size([1, 1, 128])
        #print("hidden", hidden.shape)
        output_global = self.linear_global(hidden.mean(1))
        output = self.final_activate(output_global) # torch.Size([1, 128])
        #print(output_global.shape, output.shape)
        return output
    


def model_summary():
    input_size = (1, in_size, H, W)
    model = CNN_LSTM_with_atten(inc=in_size, num_cls=CLS, embed_size= embed_size).to(device)
    #main(model, input_size)
    summary(model, input_size=input_size)

def main(model, input_size):
    inp1 = torch.rand(input_size, dtype=torch.float32).to(device)
    output = model(inp1)
    print(output.shape)


