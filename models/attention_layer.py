import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, dim, out_dim):
        super(SelfAttention,self).__init__()
        self.dim = out_dim
        self.qkv_weight = nn.Linear(dim, out_dim*3, bias=False)
    
    def forward(self, x):
        q, k, v = self.qkv_weight(x).chunk(3, dim=-1)
        att_logit = torch.bmm(q, k.transpose(1,2)) * (self.dim ** -0.5) # q * k
        att_weight = torch.softmax(att_logit, dim=-1)
        weighted_v = torch.bmm(att_weight, v) # q*k@k*dim == q * dim
        return weighted_v


class MultiheadSelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim, heads):
        super(MultiheadSelfAttention,self).__init__()
        head_out = out_dim // heads
        self.heads = nn.ModuleList([SelfAttention(in_dim, head_out) for _ in range(heads)])
    def forward(self, x):
        outs = []
        for head in self.heads:
            out = head(x)
            outs.append(out)
        outs = torch.cat(outs, dim=2)
        return outs 

