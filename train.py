import logging
from pathlib import Path
import sys, os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from cfg import Cfg
from utils.data_loder import MetaDataLoader
from models import LSTM_with_atten
from utils import utils
from utils.optimizer import create_optimizer
from utils.scheduler import CosineWithRestarts

H = W = 224

class Trainer:
    def __init__(self, config):
        self.tfwriter = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR)
        self.criterion = nn.NLLLoss() 

    def call_data_loader(self, config, num_worker):
        """ Dataset And Augmentation
        if -1 ~ 1 normalize, use two:
        transforms.ToTensor() 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if 0 ~ 1 normalize, just use:
        transforms.ToTensor()
        """
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=5),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

        val_transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        self.train_dst = MetaDataLoader(config, transform=train_transform, valid=None)
        self.val_dst = MetaDataLoader(config, transform=val_transform, valid=True)
        
        
        print("Train set: %d, Val set: %d" %(len(self.train_dst), len(self.val_dst)))
        train_loader = data.DataLoader(self.train_dst, batch_size=config.train_batch, shuffle=True, num_workers=num_worker, pin_memory=True)
        val_loader = data.DataLoader(self.val_dst, batch_size=config.val_batch, shuffle=True, num_workers=0, pin_memory=None)

        return train_loader, val_loader


    def validate(self, val_loader, model, global_step, epoch, device):
        interval_valloss = 0
        nums = 0
        model.eval()
        print("validating .....")
        with torch.no_grad():
            for i, (val_x1, val_x2, val_y) in tqdm(enumerate(val_loader)):
                val_x1 = val_x1.to(device=device, dtype=torch.float32)
                val_x2 = val_x2.to(device=device, dtype=torch.float32)
                val_y = val_y.to(device=device, dtype=torch.long)
                # predict valid
                val_pred = model(val_x1, val_x2)
                
                # val loss update
                val_loss = self.criterion(val_pred, val_y)
                interval_valloss += val_loss.item()
                nums += 1

            self.tfwriter.add_scalar('valid/interval_loss', interval_valloss/nums, global_step)
            print("Epoch %d, Itrs %d, valid_Loss=%f" % (epoch, global_step, interval_valloss/nums))
    
    
    def train(self, config, device, num_worker, weight_path=None):
        train_loader, val_loader = self.call_data_loader(config, num_worker)
        model = LSTM_with_atten(config, inc=config.channels, start_fm=config.start_fm, num_cls=config.classes, embed_size=config.embed_size)
        
        # (Initialize logging)
        logging.info(f'''Starting training:
            Epochs:          {config.epochs}
            Learning rate:   {config.lr}
            Training size:   {len(self.train_dst)}
            Validation size: {len(self.val_dst)}
        ''')
        
        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        optimizer = create_optimizer(model, config)
        scheduler = CosineWithRestarts(optimizer, t_max=10)
        #scheduler = CosineAnnealingLR(optimizer, T_max=config.t_max, eta_min=config.eta_min)
        if weight_path is not None:
            model.load_state_dict(torch.load(weight_path, map_location=device))
        #if torch.cuda.device_count() > 1:
           # model = nn.DataParallel(model)
        model.to(device)

        global_step = 0
        # 5. Begin training
        for epoch in range(1, config.epochs+1):
            interval_loss = 0
            model.train()
            with tqdm(total=int(len(self.train_dst)/config.train_batch), desc=f'Epoch {epoch}/{config.epochs}') as pbar:
                for x1, x2, y in train_loader:
                    x1 = x1.to(device=device, dtype=torch.float32)
                    x2 = x2.to(device=device, dtype=torch.float32)
                    y_label = y.to(device=device, dtype=torch.long)
                    # predict
                    pred_label = model(x1, x2)
                    #print(x_img.shape, y_label.shape, pred_label.shape)
                    #if global_step % 10==0:
                    #    utils.save_in_progress(x1, x2, global_step)
                    #print(x2.max(), x2.min(), x1.max(), x1.min(), torch.sum(pred_label))
                    # loss update
                    loss = self.criterion(pred_label, y_label)
                    interval_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                    pbar.update()
                    global_step += 1
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # Evaluation round
                    if global_step % 10 == 0:
                        self.tfwriter.add_scalar('train/train_loss', interval_loss/10, global_step)
                        print("Epoch %d, Itrs %d, Loss=%f" % (epoch, global_step, interval_loss/10))
                        interval_loss = 0.0

                    if global_step % config.val_interval == 0:
                        self.validate(val_loader, model, global_step, epoch, device)

            if config.save_checkpoint:
                Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), config.ckpt_dir + "/"+ 'checkpoint_epoch{}.pth'.format(epoch))
                logging.info(f'Checkpoint {epoch} saved!')


if __name__ == '__main__':
    cfg = Cfg
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    num_workers=1
    weight_path = None
    trainer = Trainer(cfg)
    try:
        trainer.train(cfg, device=device, num_worker=num_workers, weight_path=weight_path)
    except KeyboardInterrupt:
        logging.info('Saved interrupt')
        raise
