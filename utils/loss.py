import torch
from torch import nn

def one_hot_loss(logits, target, num_cls):
    criterion = nn.L1Loss()
    binary_coross = nn.BCELoss()
    # y target
    target = nn.functional.one_hot(target, num_classes=num_cls)
    # losses
    mae_loss = criterion(logits, target/num_cls)
    bce_loss = binary_coross(logits, target/num_cls)
    return mae_loss + bce_loss

