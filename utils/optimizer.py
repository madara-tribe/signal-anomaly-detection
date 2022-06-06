import torch
import torch.optim as optim
from .scheduler import CosineWithRestarts

def create_optimizer(model, config):
    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = torch.optim.Adam(params=[
        {'params': model.parameters(), 'lr': 0.1*config.lr},
        ], lr=config.lr, betas=(0.9, 0.999), eps=1e-08)
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = torch.optim.SGD(params=[
        {'params': model.parameters(), 'lr': 0.1*config.lr},
        ], lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay, nesterov=True)
        
    return optimizer
