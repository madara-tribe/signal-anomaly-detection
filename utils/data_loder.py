import os
import sys
import torch
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image

W = H = 224

class BasicDataLoader(data.Dataset):
    def __init__(self,
                 x_img, y_meta,
                 width: int = W,
                 height: int = H,
                 val=None):
        self.width = width
        self.height = height
        # train image
        x_imgs = os.listdir(x_img)
        x_imgs.sort()
        x_imgs = [os.path.join(x_img, path) for path in x_imgs]
        y_metas = os.listdir(y_meta)
        y_metas.sort()
        y_metas = [os.path.join(y_meta, path) for path in y_metas]
        if val:
            self.y_metas = y_metas[100:150]
            self.x_imgs = x_imgs[100:150]
        else:
            self.y_metas = y_metas[:100]
            self.x_imgs = x_imgs[:100]
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
        x_img = cv2.imread(self.x_imgs[index])
        #x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        x_img = x_img.transpose(2, 0, 1).astype(np.float32)/255
        x_img = torch.as_tensor(x_img.copy()).float().contiguous()

        y_meta = cv2.imread(self.y_metas[index])
        y_meta = cv2.cvtColor(y_meta, cv2.COLOR_BGR2LAB)
        y_meta = y_meta.transpose(2, 0, 1).astype(np.float32)/255
        y_meta = torch.as_tensor(y_meta.copy()).float().contiguous()

        return x_img, y_meta


class MetaDataLoader(data.Dataset):
    def __init__(self,
                 image_dir, mask_dir,
                 width=W,
                 height=H,
                 transform=None,
                 valid=None):
        # train image
        self.height = H
        self.width = W
        self.transform = transform
        x_imgs = os.listdir(image_dir)
        x_imgs.sort()
        x_imgs = [os.path.join(image_dir, path) for path in x_imgs]
        y_metas = os.listdir(mask_dir)
        y_metas.sort()
        y_metas = [os.path.join(mask_dir, path) for path in y_metas]
        if valid:
            self.y_metas = y_metas[200:220]
            self.x_imgs = x_imgs[200:220]
        else:
            self.y_metas = y_metas[:100]
            self.x_imgs = x_imgs[:100]
        assert (len(self.y_metas) == len(self.x_imgs))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        image = Image.open(self.x_imgs[index])
        image = image.convert('RGB')
        image = image.resize((self.height, self.width))
        masks = Image.open(self.y_metas[index])
        masks = masks.convert('HSV')
        masks = masks.resize((self.height, self.width))
        if self.transform is not None:
            image = self.transform(image)
            masks = self.transform(masks)
       
        return image, masks
        
    def __len__(self):
        return len(self.x_imgs)


