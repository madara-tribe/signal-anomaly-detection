import os
import sys
import torch
import torch.utils.data as data
import numpy as np
import cv2


class MetaDataLoader(data.Dataset):
    def __init__(self, config, 
                 transform=None,
                 valid=None):
        self.width = config.width
        self.height = config.height
        self.classes = config.classes
        self.transform = transform
        # train image
        x_img_, x2, x3 = config.x_img, config.x2, config.x3
        x_imgs = os.listdir(x_img_)
        x_imgs.sort()
        x_imgs = [os.path.join(x_img_, path) for path in x_imgs]
        x2_ = os.listdir(x2)
        x2_.sort()
        x2_ = [os.path.join(x2, path) for path in x2_]
        x3_ = os.listdir(x3)
        x3_.sort()
        x3_ = [os.path.join(x2, path) for path in x3_]
        if valid:
            N =20
            self.y_label = np.load(config.y)[:N]
            self.x_imgs, self.x2, self.x3 = x_imgs[:N], x2_[:N], x3_[:N]
        else:
            N = 200
            self.y_label = np.load(config.y)[:N]
            self.x_imgs, self.x2, self.x3 = x_imgs[:N], x2_[:N], x3_[:N]
        assert (len(self.y_label) == len(self.x_imgs))
        assert (len(self.y_label) == len(self.x2))

    def __len__(self):
        return len(self.x_imgs)

    def preprocess(self, p, clannel):
        if clannel==3:
            x = cv2.imread(p)
            x = x.reshape(self.width, self.height, 3).astype(np.float32)
        elif clannel==1:
            x = cv2.imread(p, 0)
            x = x.reshape(self.width, self.height, 1).astype(np.float32)
            #x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        return x/255
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        x_img = self.preprocess(self.x_imgs[index], 3)

        x2 = self.preprocess(self.x2[index], 1)
        x3 = self.preprocess(self.x3[index], 1)
        
        x_meta = np.concatenate([x2, x3], 2)
        x_meta = torch.as_tensor(x_meta.reshape(2, self.width, self.height).copy()).float().contiguous()
        #x_img = x_img.reshape(self.width, self.height, 3).copy()
        
        y_labels = self.y_label[index]
        y_labels = torch.as_tensor(y_labels.copy())
        
        if self.transform is not None:
            x_img = self.transform(x_img)
        return x_img, x_meta, y_labels




