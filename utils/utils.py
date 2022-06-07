import os
import cv2
import numpy as np

def ToGray(im):
    im_gray = 0.299 * im[:, :, 2] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 0]
    return im_gray

def create_binary_mask(pred, threshold=0.1):
    markers = np.zeros_like(pred)
    markers[pred < threshold] = 0
    markers[pred > threshold] = 1
    #print(np.unique(markers))
    return markers

def save_in_progress(img1, img2, nums):
    img1 = img1[0].detach().cpu().numpy()
    img1 = (img1 * 255).transpose(1, 2, 0)
    img2 = img2[0].detach().cpu().numpy()
    img2 = (img2 * 255).transpose(1, 2, 0)
    img = np.hstack([img1, img2])
    os.makedirs('InProgress', exist_ok=True)
    cv2.imwrite("InProgress/pred_rgb_{}.png".format(nums), img.astype(np.uint8))

