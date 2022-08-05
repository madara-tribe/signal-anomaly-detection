import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.train_batch = 2
Cfg.val_batch = 1
Cfg.lr = 0.001
Cfg.epochs = 50
Cfg.val_interval = 500
Cfg.gpu_id = '1'
Cfg.weight_decay = 1e-4
Cfg.momentum = 0.9
#Cfg.TRAIN_OPTIMIZER = 'sgd'
Cfg.TRAIN_OPTIMIZER = 'adam'
Cfg.channel=30000
Cfg.tstep=3
Cfg.hidden_dim=256 # Encoder output dim
Cfg.embed_dim=256 # CNN Encoder first embed size
## dataset
Cfg.x = "data/X.npy"
Cfg.y = "data/y_label.npy"
#Cfg.aug_x = "data/X_aug.npy"
Cfg.aug_x = "data/tshift.npy"
Cfg.val_x = "data/val_X.npy"
Cfg.val_y = "data/val_y.npy"
Cfg.save_checkpoint = True
Cfg.TRAIN_TENSORBOARD_DIR = './logs'
Cfg.ckpt_dir = os.path.join(_BASE_DIR, 'checkpoints')


