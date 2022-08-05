import numpy as np
from scipy import signal

def apply_bandpass(x, lf=1, hf=100, order=16, sr=30000):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    x = signal.sosfiltfilt(sos, x) / normalization
    return x

def preprocess(x, tstep, sr):
    for i in range(tstep):
        if len(x.shape)==3:
            x_ = x[:, :, i]
            x_ = x_ / np.max(np.abs(x_), axis=-1, keepdims=True)
            x_ *= signal.tukey(sr, 0.1)
            x[:, :, i] = apply_bandpass(x_, sr=sr)
        elif len(x.shape)==2:
            x_ = x[:, i]
            x_ = x_ / np.max(np.abs(x_), axis=-1, keepdims=True)
            x_ *= signal.tukey(sr, 0.1)
            x[:, i] = apply_bandpass(x_, sr=sr)
    return x
