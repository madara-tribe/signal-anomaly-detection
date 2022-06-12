import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.width = 224
Cfg.height = 224
Cfg.channels = 3
Cfg.train_batch = 4
Cfg.val_batch = 1
Cfg.lr = 0.001
Cfg.epochs = 30
Cfg.val_interval = 400
Cfg.gpu_id = '3'
Cfg.weight_decay = 1e-4
Cfg.momentum = 0.9
#Cfg.TRAIN_OPTIMIZER = 'sgd'
Cfg.TRAIN_OPTIMIZER = 'adam'
Cfg.classes=11
Cfg.start_fm = 64
Cfg.embed_size = 512 # CNN Encoder outut size
## Attention
Cfg.SELFATTENTION =True
Cfg.ATTENTION = None
## dataset
Cfg.x_img = "data/x_img"
Cfg.x2 = "data/x2"
Cfg.x3 = "data/x3"
Cfg.y1 = "data/y_color.npy"
Cfg.y2 = "data/y_shape.npy"
Cfg.save_checkpoint = True
Cfg.TRAIN_TENSORBOARD_DIR = './logs'
Cfg.ckpt_dir = os.path.join(_BASE_DIR, 'checkpoints')


