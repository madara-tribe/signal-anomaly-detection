import matplotlib.pyplot as plt
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

def save_in_progress(img, mask, pred, nums):
    img = img[0].detach().cpu().numpy()
    img = (img * 255).transpose(1, 2, 0)

    mask = mask[0].detach().cpu().numpy()
    mask = (mask * 255).transpose(1, 2, 0)
    pred = pred[0].detach().cpu().numpy()
    pred = (pred * 255).transpose(1, 2, 0)
    save_img = np.hstack([img, mask, pred])
    os.makedirs('InProgress', exist_ok=True)
    cv2.imwrite("InProgress/pred_rgb_{}.png".format(nums), save_img.astype(np.uint8))


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[1, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

