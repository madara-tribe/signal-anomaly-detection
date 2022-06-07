import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.width = 224
Cfg.height = 224
Cfg.channels = 1
Cfg.train_batch = 1
Cfg.val_batch = 1
Cfg.lr = 0.001
Cfg.epochs = 30
Cfg.val_interval = 200
Cfg.gpu_id = '3'
Cfg.weight_decay = 1e-4
Cfg.momentum = 0.9
#Cfg.TRAIN_OPTIMIZER = 'sgd'
Cfg.TRAIN_OPTIMIZER = 'adam'
Cfg.classes=128
Cfg.start_fm = 64
Cfg.embed_size = 1024 # CNN Encoder outut size
## Attention
Cfg.ATTENTION=True
Cfg.SELFATTENTION = True
## dataset
Cfg.x1 = "data/x1.npy"
Cfg.x2 = "data/x2.npy"
Cfg.y = "data/y_label.npy"
Cfg.save_checkpoint = True
Cfg.TRAIN_TENSORBOARD_DIR = './logs'
Cfg.ckpt_dir = os.path.join(_BASE_DIR, 'checkpoints')

