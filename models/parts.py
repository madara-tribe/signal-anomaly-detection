import torch
from torch import nn
from torch.nn import functional as F

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


