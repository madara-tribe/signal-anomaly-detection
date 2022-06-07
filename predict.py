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
    val_dst = MetaDataLoader(config, config.x1, config.x2, config.y, transform=val_transform, valid=True)
    val_loader = data.DataLoader(
                val_dst, batch_size=config.val_batch, shuffle=None, num_workers=0, pin_memory=None)
    print("Val set: %d" % len(val_dst))
    return val_loader


def predict(config, device, path):
    model = LSTM_with_atten(config, inc=config.channels, start_fm=config.start_fm, num_cls=config.classes, embed_size=config.embed_size)
    model.to(device)
    model.load_state_dict(torch.load(path))
    print("loaded trained model")
    val_loader = create_data_loader(config, transform_=True)
    nums, acc = 0, 0
    model.eval()
    print("predicting  .....")
    with torch.no_grad():
        for i, (val_x1, val_x2, val_y) in tqdm(enumerate(val_loader)):
            val_x1 = val_x1.to(device=device, dtype=torch.float32)
            val_x2 = val_x2.to(device=device, dtype=torch.float32)
            val_y = val_y.to(device=device, dtype=torch.long)
            # predict valid
            val_pred = model(val_x1, val_x2)
            
            val_pred = val_pred.detach().cpu().numpy()
            val_y = val_y.detach().cpu().numpy()
            pred_idx = np.argmax(val_pred)
            if val_y==int(pred_idx):
                acc += 1
            else:
                acc += 0
            nums += 1
        print('accuracy is {}'.format(acc/nums))

if __name__=="__main__":
    path = "checkpoints/checkpoint_epoch30.pth" #str(sys.argv[0])
    cfg = Cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict(cfg, device, path)

