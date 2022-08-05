import os, sys
sys.path.append("../")
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
from utils.audio_augmentation import audio_augmentation_apply
R=30000
def make_meta_label(df):
    zero = np.zeros((10))
    one = np.ones((10))
    L, val = [], []
    c = 0
    for i, v in tqdm(df.iterrows()):
        if v["scratchpos_edge"]!=0 or v["scratchpos_center"]!=0 or v["scratchpos_baffle"]!=0:
            label = 0
            c += 1
        else:
            label = 1
        if i < len(df)-200:
            L.append(label)
        else:
            val.append(label)
    print("normal is {}, anomaly is {} ".format(len(df)-c, c))
    return np.array(L), np.array(val)

def Tolabel(meta="train_meta.csv"):
    df = pd.read_csv(meta)
    df = df.fillna(0)
    ys, val_y = make_meta_label(df)
    print(np.unique(ys), ys.shape, val_y.shape)
    np.save("y_label", ys)
    np.save("val_y", val_y)

def prepro(df):
    dlist = ["F1-X", "F1-Y", "F2-X", "F2-Y", "F3-X", "F3-Y", "MIX1-X", "MIX1-Y"]
    # F2 as D
    df["F2-X"] = df["F2-X"]-df["F1-X"]
    df["F2-Y"] = df["F2-Y"]-df["F1-Y"]
    df["F2-Z"] = 2*(df["F2-X"] + df["F2-Y"]) - np.sqrt(df["F2-X"]**2 + df["F2-Y"]**2) # D
    
    # F3 as A
    df["F3-X"] = df["F3-X"]-df["F1-X"]
    df["F3-Y"] = df["F3-Y"]-df["F1-Y"]
    df["F3-Z"] = np.sqrt(df["F3-X"]**2 + df["F3-Y"]**2) # D

    # as S
    df["MIX"] = np.sqrt(df["MIX1-X"]**2 + df["MIX1-Y"]**2 + df["F2-Z"]**2 + df["F3-Z"]**2)
    
    # To Sparse Vector
    MEDIAN = 7000
    df.loc[(df['F2-Z'] > MEDIAN) | (df['F2-Z'] < -MEDIAN), 'F2-Z'] = 0
    df.loc[df['F3-Z'] > MEDIAN, 'F3-Z'] = 0
    df.loc[df['MIX'] > MEDIAN, 'MIX'] = 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df.drop(dlist, axis=1).astype(float)


def main():
    resize = R
    path = "train"
    flist = os.listdir(path)
    flist.sort()
    X, val = [], []
    data_size = len(flist)
    for i, fs in tqdm(enumerate(flist)):
        df = pd.read_csv(os.path.join(path, fs))
        df = prepro(df)
        xs = signal.resample(df.values, resize)
        if i < data_size-200:
            X.append(xs)
        else:
            val.append(xs)
    X, val = np.array(X), np.array(val)
    noise = np.random.normal(0, 2, X.shape)
    X_aug = X + noise
    print(X.shape, val.shape, X_aug.shape)
    np.save("X", X)
    np.save("val_X", val)
    np.save("X_aug", X_aug)

def timeshift(path="train"):
    resize = R
    flist = os.listdir(path)
    flist.sort()
    X_aug = []
    data_size = len(flist)
    for i, fs in tqdm(enumerate(flist)):
        df = pd.read_csv(os.path.join(path, fs))
        df = prepro(df)
        xs = audio_augmentation_apply(df, sr=len(df), types='timeshift')
        xs = signal.resample(xs, resize)
        if i < data_size-200:
            X_aug.append(xs)
    X_aug = np.array(X_aug)
    print(X_aug.shape)
    np.save('tshift', X_aug)


if __name__=="__main__":
    main()
    Tolabel(meta="train_meta.csv")
    #timeshift()
    #create_test(path = "test")
