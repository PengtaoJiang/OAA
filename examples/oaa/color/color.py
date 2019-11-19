import numpy as np
from PIL import Image
import os
import sys
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

cats = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']

colormaps = ['#000000', '#7F0000', '#007F00', '#7F7F00', '#00007F', '#7F007F', '#007F7F', '#7F7F7F', '#3F0000', '#BF0000', '#3F7F00',
                        '#BF7F00', '#3F007F', '#BF007F', '#3F7F7F', '#BF7F7F', '#003F00', '#7F3F00', '#00BF00', '#7FBF00', '#003F7F']

def colormap(index):
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', [colormaps[0], colormaps[index+1], '#FFFFFF'], 256)
    

def color(im_name, att_name, save_name, im_label):            
    # read origin image
    img = cv2.imread(im_name)
    img = np.array(img, dtype=np.float32)
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # read attention map
    att = cv2.imread(att_name, 0)
    att = np.array(att, dtype=np.float32)

    # combine them
    out = im_gray * 0.2 + att * 0.7
    plt.imsave(save_name, out, cmap=colormap(im_label))
    #cv2.imwrite(out_name, out)

if __name__ == '__main__':
    im_name = './2007_000121.jpg'
    att_name = './2007_000121_19.png'
    save_name = './2007_000121_19_color.png'
    im_label = 19
    color(im_name, att_name, save_name, im_label)