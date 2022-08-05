import torch.nn as nn
import torch.nn.functional as F
import torch

class RobustLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super(RobustLoss, self).__init__()
        self.nnloss = nn.NLLLoss()
        self.lce = LabelSmoothingCrossEntropy(epsilon=epsilon)
    def forward(self, sfm, logsfm, target):
        nn_loss_ = self.nnloss(logsfm, target)
        lce_ = self.lce(sfm, target)
        return (nn_loss_ + lce_)/2


def linear_combination(x, y, epsilon):
    return (1 - epsilon) * x + epsilon * y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(nll, loss/n, self.epsilon)
