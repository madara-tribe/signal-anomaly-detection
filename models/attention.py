import torch
from torch import nn
from torch.nn import functional as F


def attention_net(lstm_output, final_state):
    lstm_output = lstm_output.permute(1, 0, 2) # keys
    querys = final_state.squeeze(0) # query
    logits = torch.bmm(lstm_output, querys.unsqueeze(2)).squeeze(2)
    attn_weights = F.softmax(logits, dim=1)
    new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), # value
                                 attn_weights.unsqueeze(2)).squeeze(2)
    return new_hidden_state

class SelfAttention(nn.Module):
    def __init__(self, lstm_dim, num_cls):
        super(SelfAttention, self).__init__()
        self.lstm_dim = lstm_dim *2
        self.num_cls = num_cls
        self.attn_weight = nn.Sequential(
            nn.Linear(lstm_dim *2, lstm_dim *2), 
            nn.Tanh(),
            nn.Linear(lstm_dim *2, lstm_dim *2)
        )
    def forward(self, out):
        attention_weight = F.softmax(self.attn_weight(out), dim=1)
        feats = torch.add(out, attention_weight)
        return feats
