
import sys, os
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
from cfg import Cfg
from utils.data_loder import MetaDataLoader
from models import LSTM_with_atten

val_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def create_data_loader(config, transform_=None):
    val_dst = MetaDataLoader(config, transform=val_transform, valid=None)
    val_loader = data.DataLoader(
                val_dst, batch_size=config.val_batch, shuffle=None, num_workers=0, pin_memory=None)
    print("Val set: %d" % len(val_dst))
    return val_loader


def predict(config, device, path):
    model = LSTM_with_atten(config, inc=config.channels, start_fm=config.start_fm, num_cls=config.classes, 
                      embed_size=config.embed_size, lstm_use=None)
    model.to(device)
    model.load_state_dict(torch.load(path))
    print("loaded trained model")
    val_loader = create_data_loader(config, transform_=True)
    nums, cacc, sacc = 0, 0, 0
    model.eval()
    print("predicting  .....")
    with torch.no_grad():
        for val_x1, val_x2, val_y1, val_y2 in tqdm(val_loader):
            val_x1 = val_x1.to(device=device, dtype=torch.float32)
            val_x2 = val_x2.to(device=device, dtype=torch.float32)
            val_y1 = val_y1.to(device=device, dtype=torch.long)
            val_y2 = val_y2.to(device=device, dtype=torch.long)
            
            # predict valid
            val_color, val_shape = model(val_x1, val_x2)
           
            # calor 
            val_color = val_color.detach().cpu().numpy()
            val_y1 = val_y1.detach().cpu().numpy()
            cacc += 1 if val_y1==np.argmax(val_color) else 0
            # shape
            val_shape = val_shape.detach().cpu().numpy()
            val_y2 = val_y2.detach().cpu().numpy()
            sacc += 1 if val_y2==np.argmax(val_shape) else 0
            nums += 1
        print("shape accuracy is {}".format(sacc/nums))
        print('color accuracy is {}'.format(cacc/nums))
     

if __name__=="__main__":
    idx = int(sys.argv[1])
    path = "checkpoints/checkpoint_epoch{}.pth".format(int(idx))
    cfg = Cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict(cfg, device, path)



