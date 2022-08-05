import logging
from pathlib import Path
import sys, os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from torch.optim.lr_scheduler import CosineAnnealingLR
from cfg import Cfg
from utils.data_loder import DataLoader
from models.model import CNN1d_Transformer
from utils import optimizer, scheduler, losses

class Trainer:
    def __init__(self, config, device):
        self.tfwriter = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR)
        self.criterion = losses.RobustLoss()
        self.tstep = config.tstep
        self.c = config.channel
        self.device = device
    def call_data_loader(self, config, num_worker):
        self.train_dst = DataLoader(config, transform=None, valid=None)
        self.val_dst = DataLoader(config, transform=None, valid=True)
        
        print("Train set: %d, Val set: %d" %(len(self.train_dst), len(self.val_dst)))
        train_loader = data.DataLoader(self.train_dst, batch_size=config.train_batch, shuffle=True, num_workers=num_worker, pin_memory=True)
        val_loader = data.DataLoader(self.val_dst, batch_size=config.val_batch, shuffle=True, num_workers=0, pin_memory=None)

        return train_loader, val_loader

    
    def load_model(self, config, weight_path):
        model = CNN1d_Transformer(config.tstep, config.embed_dim, config.hidden_dim)
        #summary(model, (1, self.c, config.tstep))
        self.opt = optimizer.create_optimizer(model, config)
        self.scheduler_ = scheduler.CosineWithRestarts(self.opt, t_max=10)
        #scheduler = scheduler.CosineAnnealingLR(optimizer_, T_max=config.t_max, eta_min=config.eta_min)
        if weight_path is not None:
            model.load_state_dict(torch.load(weight_path))
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(self.device)
        return model

    def validate(self, config, val_loader, model, global_step, epoch):
        interval_valloss =0
        nums = 0
        model.eval()
        print("validating .....")
        with torch.no_grad():
            for val_x, val_y in tqdm(val_loader):
                val_x = val_x.to(device=self.device, dtype=torch.float32)
                val_y = val_y.to(device=self.device, dtype=torch.long)
                sfm, logsfm = model(val_x)
                #val_y = val_y.reshape(-1, 1)
                # val loss update
                val_loss = self.criterion(sfm, logsfm, val_y)
                interval_valloss += val_loss.item()
                nums += 1
            self.tfwriter.add_scalar('valid/interval_loss', interval_valloss/nums, global_step)
            print("Epoch %d, Itrs %d, valid_Loss=%f" % (epoch, global_step, interval_valloss/nums))
        return interval_valloss/nums
    
    def train(self, config, num_worker, weight_path=None):
        train_loader, val_loader = self.call_data_loader(config, num_worker)
        model = self.load_model(config, weight_path)
        
        # (Initialize logging)
        logging.info(f'''Starting training:
            Epochs:          {config.epochs}
            Device:          {self.device}
            Learning rate:   {config.lr}
            Training size:   {len(self.train_dst)}
            Validation size: {len(self.val_dst)}
        ''')
        best_score = 10000
        global_step = 0
        for epoch in range(1, config.epochs+1):
            interval_loss = 0
            model.train()
            with tqdm(total=int(len(self.train_dst)/config.train_batch), desc=f'Epoch {epoch}/{config.epochs}') as pbar:
                for x, y in train_loader:
                    x = x.to(device=self.device, dtype=torch.float32)
                    y = y.to(device=self.device, dtype=torch.long)
                    # predict
#                    np.save("npy/fftx{}".format(global_step), x.to('cpu').detach().numpy().copy())
                    sfm, logsfm = model(x)
                    #y = y.reshape(-1, 1)
                    loss_ = self.criterion(sfm, logsfm, y)
                    interval_loss += loss_.item()
                    loss_.backward()
                    self.opt.step()

                    pbar.update()
                    global_step += 1
                    pbar.set_postfix(**{'loss (batch)': loss_.item()})

                    if global_step % 10 == 0:
                        self.tfwriter.add_scalar('train/train_loss', interval_loss/10, global_step)
                        print("Epoch %d, Itrs %d, Loss=%f" % (epoch, global_step, interval_loss/10))
                        interval_loss = 0.0

                    if global_step % config.val_interval == 0:
                        val_loss = self.validate(config, val_loader, model, global_step, epoch)
                        if val_loss < best_score:
                            best_score = val_loss
                            Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)
                            torch.save(model.state_dict(), config.ckpt_dir + "/"+ 'checkpoint_epoch{}_{}.pth'.format(epoch, np.round(best_score, decimals=4)))

                    model.train()
                model.train()
            self.scheduler_.step()


if __name__ == '__main__':
    cfg = Cfg
    gpu=cfg.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    num_workers=1
    if len(sys.argv)>0:
        weight_path = str(sys.argv[1])
    else:
        weight_path = None
    trainer = Trainer(cfg, device)
    try:
        trainer.train(cfg, num_worker=num_workers, weight_path=weight_path)
    except KeyboardInterrupt:
        logging.info('Saved interrupt')
        raise




