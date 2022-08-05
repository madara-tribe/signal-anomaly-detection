import os
import torch
import torch.utils.data as data
import numpy as np
from .signal_utils import preprocess
from .audio_augmentation import audio_augmentation_apply

class DataLoader(data.Dataset):
    def __init__(self, config, 
                 transform=None,
                 valid=None):
        self.channels = config.channel
        self.tstep = config.tstep
        self.transform = transform
        self.valid = valid
        # train image
        if not self.valid:
            x, y = np.load(config.x), np.load(config.y)
            aug_x = np.load(config.aug_x) #, np.load(config.aug_x1)
            self.x = np.concatenate([x, aug_x], 0)
            self.y = np.concatenate([y, y], 0)
            print("f", self.x.shape, len(self.x), len(self.y))
        else:
            self.x, self.y = np.load(config.val_x), np.load(config.val_y)
        assert (len(self.x) == len(self.y))

    def __len__(self):
        return len(self.x)

    def preprocess(self, x):
        """
        signal data must not reshape. shape must be (batch, channel, time_step)
        ex: (1, 30000, 6)
        """
        #x = preprocess(x, self.tstep, sr=self.channels)
        #x = np.fft.fft(x)
        x = x / np.max(x)
        return x

        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        x = self.preprocess(self.x[index].reshape(self.channels, self.tstep))
        
        y = self.y[index] #.iastype(np.float32)
        
        if self.transform is not None:
            pass
        return x.astype(np.float32), y
