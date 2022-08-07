import sys, os
sys.path.append('../')
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from scipy import signal
import torch
import torch.nn as nn
from cfg import Cfg
from models.model import CNN1d_LSTM
from tqdm import tqdm

def df_preproccess(df):
    dlist = ["F1-X", "F1-Y", "F2-X", "F2-Y", "F3-X", "F3-Y", "MIX1-Y", "pos"]
    # F2 as D
    df["F2-X"] = (df["F2-X"]-df["F1-X"])
    df["F2-Y"] = (df["F2-Y"]-df["F1-Y"])
    df["F2-Z"] = 2*(df["F2-X"] + df["F2-Y"]) - np.sqrt(df["F2-X"]**2 + df["F2-Y"]**2) # D

    # F3 as A
    df["F3-X"] = df["F3-X"]-df["F1-X"]
    df["F3-Y"] = df["F3-Y"]-df["F1-Y"]
    df["F3-Z"] = np.sqrt(df["F3-X"]**2 + df["F3-Y"]**2) # D

    # as S
    df["MIX"] = np.sqrt(df["MIX1-X"]**2 + df["MIX1-Y"]**2 + df["F2-Z"]**2 + df["F3-Z"]**2)
    df["MIX1-X"] = np.sqrt(df['MIX1-X']**2)

    # To Sparse Vector
    MEDIAN = 1000
    df.loc[(df['F2-Z'] > MEDIAN) | (df['F2-Z'] < -MEDIAN), 'F2-Z'] = 0
    df.loc[df['F3-Z'] > MEDIAN, 'F3-Z'] = 0
    df.loc[df['MIX'] > MEDIAN, 'MIX'] = 0
    df.loc[df['MIX1-X'] > MEDIAN, 'MIX1-X'] = 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df.drop(dlist, axis=1).astype(float)
 
def load_model(config, path, device):
    model = CNN1d_LSTM(config.tstep, config.embed_dim, config.hidden_dim)
    model.load_state_dict(torch.load(path))
    if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    return model

def load_csvs():
    csv_paths = [str(x) for x in list(glob.glob("test/*.csv") )]
    files = [x.split("/")[-1] for x in csv_paths]

    test_csv = pd.DataFrame({
        "filename":files,
        "filepath":csv_paths,
    })
    sample_submission = pd.read_csv("sample_submission.csv")
    return test_csv, sample_submission

idx = 28
path = "../checkpoints/checkpoint_epoch{}.pth".format(int(idx))
cfg = Cfg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(cfg, path, device)
model.eval()

test_csv, sample_submission = load_csvs()
filepaths = test_csv["filepath"].values
filenames = test_csv["filename"].values

tq_ = tqdm(zip(filepaths,filenames), total=len(filepaths))
size = 30000
th = 0.8
sub_preds=[]
sub_ids=[]
for i,(path,name) in enumerate(tq_):
    filename = path.split("/")[-1]
    test = pd.read_csv(path)
    name = name[:-4]
    
    test = test.reset_index(drop=False).rename(columns={"index":"pos"})
    test = df_preproccess(test)
    x = signal.resample(test.values, size)
    x = x / np.max(x)
    x = x.reshape(1, size, 4)
    #np.save("npy/{}wav".format(i), x.astype(np.float32))
    x = torch.from_numpy(x.astype(np.float32)).clone()
    pred, _ = model(x)
    pred = pred[0].to('cpu').detach().numpy().copy()
    test_pred = 1 if pred >= th else 0
    num_50 = len(sample_submission[sample_submission["id"].str.contains(name)])
    sub_ids = sub_ids + [name+ "_" +str(50*(x+1)) for x in range(num_50)]
    sub_preds = sub_preds + [test_pred for x in range(num_50)]
#    print(test_pred)
submission = pd.DataFrame({
    "id":sub_ids,
    "target":sub_preds
})
submission.to_csv("submission_.csv",index=False)
