import os
import sys
import torch
import torch.utils.data as data
import numpy as np
import cv2


class MetaDataLoader(data.Dataset):
    def __init__(self, config, 
                 x1, x2, y,
                 transform=None,
                 valid=None):
        self.width = config.width
        self.height = config.height
        self.classes = config.classes
        self.transform = transform
        # train image
        #x_imgs = os.listdir(x_img)
        #x_imgs.sort()
        #x_imgs = [os.path.join(x_img, path) for path in x_imgs]
        #y_metas = os.listdir(y_meta)
        #y_metas.sort()
        #y_metas = [os.path.join(y_meta, path) for path in y_metas]
        if valid:
            self.y_label = np.load(y)[:200]
            x1 = np.load(x1)[:200]
            x2 = np.load(x2)[:200]
            self.x1, self.x2 = np.flip(x1), np.flip(x2)
        else:
            self.y_label = np.load(y)[:200]
            self.x1 = np.load(x1)[:200]
            self.x2 = np.load(x2)[:200]
        assert (len(self.y_label) == len(self.x1))
        assert (len(self.y_label) == len(self.x2))

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        x1 = self.x1[index]
        x2 = self.x2[index]
        #x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        x1 = x1.reshape(self.width, self.height, 1).astype(np.float32)/255
        x2 = x2.reshape(self.width, self.height, 1).astype(np.float32)/255
        #x_img = torch.as_tensor(x_img.copy()).float().contiguous()

        y_labels = self.y_label[index]
        y_labels = torch.as_tensor(y_labels.copy())
        #y_labels = torch.nn.functional.one_hot(y_labels, num_classes=self.classes)
        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        return x1, x2, y_labels



