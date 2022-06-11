import torch
from torch import nn

class OneHotLoss:
    def __init__(self):
        self.mae = nn.L1Loss()
        self.binary_coross = nn.BCELoss()
    def color_loss(self, logits, target, num_cls):
         # y target
         target = nn.functional.one_hot(target, num_classes=num_cls)
         mae_loss = self.mae(logits, target/2)
         bce_loss = self.binary_coross(logits, target/2)
         return mae_loss + bce_loss
    
    def shape_loss(self, logits, target, num_cls):
         # y target
         target = nn.functional.one_hot(target, num_classes=num_cls)
         mae_loss = self.mae(logits, target/2)
         bce_loss = self.binary_coross(logits, target/2)
         return mae_loss + bce_loss




