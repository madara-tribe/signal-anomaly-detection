import sys, os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data
from cfg import Cfg
from utils.data_loder import DataLoader
from models.model import CNN1d_Transformer


def create_data_loader(config, transform_=None):
    val_dst = DataLoader(config, transform=None, valid=True)
    val_loader = data.DataLoader(val_dst, batch_size=config.val_batch, shuffle=None, num_workers=0, pin_memory=None)
    print("Val set: %d" % len(val_dst))
    return val_loader


def test(config, device, path):
    model = CNN1d_Transformer(config.tstep, config.embed_dim, config.hidden_dim)
    model.load_state_dict(torch.load(path))
    if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    print("loaded trained model")
    val_loader = create_data_loader(config, transform_=True)
    nums, acc, oacc = 0, 0, 0
    th = 0.6
    model.eval()
    print("predicting  .....")
    with torch.no_grad():
        for val_x, val_y in tqdm(val_loader):
            val_x = val_x.to(device=device, dtype=torch.float32)
            #val_y = val_x.to(device=device, dtype=torch.long)
            # predict valid
            log1, log2 = model(val_x)
            logits = log1 #+ log2
            prob = logits.detach().cpu().numpy()
            print(np.argmax(prob), val_y)
            pred_idx = 0 if np.argmax(prob)==0 else 1
            ys = 0 if int(val_y) == 0 else 1
            oacc += 1 if pred_idx==ys else 0
            acc += 1 if np.argmax(prob)==int(val_y) else 0
            nums += 1
        print("accuracy is {}".format(acc/nums))
        print("other accuracy is {}".format(oacc/nums))
     

if __name__=="__main__":
    p = str(sys.argv[1])
    path = os.path.join("checkpoints", p)
    cfg = Cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test(cfg, device, path)


