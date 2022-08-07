import os, sys
sys.path.append("../")
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
from utils.audio_augmentation import audio_augmentation_apply
import glob

def get_center_ids(df_, target="MIX1-Y",  th=7500, margin=1000):
    harf_pos = int(len(df_)/2)
    
    first_th = -th
    last_th = th

    df_first = df_[:harf_pos]
    first_ids = df_first[df_first[target]< first_th].index
    

    first_max = first_ids.max()
    
    s_center = first_max + margin

    df_last = df_[harf_pos:]
    last_ids = df_last[df_last[target]> last_th].index
    
    last_min = last_ids.min()
    
    e_center = last_min - margin
    
    if len(first_ids) == 0:
        s_center = 0
    
    if len(last_ids) == 0:
        e_center = len(df_) -1

    return s_center, e_center

def create_meta(path="train_meta.csv"):
    df_meta = pd.read_csv(path)
    center_csv_paths = [str(x) for x in list(glob.glob("train/*.csv") )]
    files = [x.split("/")[-1] for x in center_csv_paths]

    df_csv = pd.DataFrame({
        "filename":files,
        "filepath":center_csv_paths,
    })
    df_meta = pd.merge(df_meta,df_csv, on="filename", how="left")
    df_meta["scratchpos_edge"] = df_meta["scratchpos_edge"].apply( lambda x: [int(y) for y in x.split()] if not pd.isna(x) else [])
    df_meta["scratchpos_center"] = df_meta["scratchpos_center"].apply( lambda x: [int(y) for y in x.split()] if not pd.isna(x) else [])
    df_meta["scratchpos_baffle"] = df_meta["scratchpos_baffle"].apply( lambda x: [int(y) for y in x.split()] if not pd.isna(x) else [])

    df_meta["num_edge"] = df_meta["scratchpos_edge"].apply( lambda x: int(len(x) / 2))
    df_meta["num_center"] = df_meta["scratchpos_center"].apply( lambda x: int(len(x) / 2))
    df_meta["num_baffle"] = df_meta["scratchpos_baffle"].apply( lambda x: int(len(x) / 2))

    df_meta["num_all"] = df_meta["num_edge"] + df_meta["num_center"] + df_meta["num_baffle"]

    def merge_scratch(row):

        merged_pos = row["scratchpos_edge"] + row["scratchpos_center"] + row["scratchpos_baffle"]
        merged_pos.sort()

        return merged_pos

    df_meta["scratchpos_all"] = df_meta.apply(merge_scratch, axis=1)
    return df_meta

def preprocess(df, med):
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
    MEDIAN = med #7000
    df.loc[(df['F2-Z'] > MEDIAN) | (df['F2-Z'] < -MEDIAN), 'F2-Z'] = 0
    df.loc[df['F3-Z'] > MEDIAN, 'F3-Z'] = 0
    df.loc[df['MIX'] > MEDIAN, 'MIX'] = 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df.drop(dlist, axis=1) #.astype(float)

R = 30000
def make_train_data(df_, med, valid=None):
    paths = df_["filepath"].values
    scratches = df_["scratchpos_all"].values
    resize = R
    Xb, valb = [], []
    yb, val_yb = [], []
    total = len(paths)
    tq_ = tqdm(zip(paths,scratches), total=len(paths))
    for i, (path, scratch) in enumerate(tq_):
        df = pd.read_csv(path)
        s, e = get_center_ids(df, margin=0)
        df = df[s:e]

        df = preprocess(df, med)
        s_scratch = scratch[0::2]
        e_scratch = scratch[1::2]
        df["target"] = 0
        if len(scratch)>0:
            for i in range(len(scratch)-1):
                if i%2==0:
                    s = scratch[i]
                    e = scratch[i+1]
                    df.loc[s:e, "target"] = 1
        y = np.array(df["target"].values, dtype=np.uint8)
        X = df.drop("target", axis=1).values
        xs = signal.resample(X, resize)
        ys = signal.resample(y, resize)
        if valid:
            if i < total-200:
                Xb.append(xs)
                yb.append(ys)
            else:
                valb.append(xs)
                val_yb.append(ys)
        else:
            Xb.append(xs)
            yb.append(ys)
    Xb, yb = np.array(Xb), np.array(yb, dtype=np.uint8)
    print(Xb.shape, yb.shape, np.unique(yb))
    np.save("X"+str(med), Xb)
    np.save("y"+str(med), yb)
    if valid:
        val, val_y = np.array(valb), np.array(val_yb, dtype=np.uint8)
        print(val.shape, val_y.shape, np.unique(val_y))
        np.save("val"+str(med), val)
        np.save("val_y"+str(med), val_y)

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

    X_aug = np.array(X_aug)
    print(X_aug.shape)
    np.save('tshift', X_aug)


if __name__=="__main__":
    med = int(sys.argv[1])
    if len(sys.argv)>2:
        valid = str(sys.argv[2])
    else:
        valid = False
    meta = create_meta(path="train_meta.csv")
    make_train_data(meta, med, valid)
    #timeshift()

