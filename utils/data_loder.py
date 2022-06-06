import os
import sys
import torch
import torch.utils.data as data
import numpy as np
import cv2


class MetaDataLoader(data.Dataset):
    def __init__(self, config, 
                 x_imgs, y_meta,
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
            self.y_metas = np.load(y_meta)
            self.x_imgs = np.load(x_imgs)
        else:
            self.y_metas = np.load(y_meta)
            self.x_imgs = np.load(x_imgs)
        assert (len(self.y_metas) == len(self.x_imgs))

    def __len__(self):
        return len(self.x_imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        x_img = self.x_imgs[index]
        #x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        x_img = x_img.astype(np.float32)/255
        #x_img = torch.as_tensor(x_img.copy()).float().contiguous()

        y_labels = self.y_metas[index]
        print(y_labels)
        y_labels = torch.as_tensor(y_labels.copy()).int().contiguous()
        #y_labels = torch.nn.functional.one_hot(y_labels, num_classes=self.classes)
        if self.transform is not None:
            x_img = self.transform(x_img)
        return x_img, y_labels


