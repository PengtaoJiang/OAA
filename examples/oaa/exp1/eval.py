import numpy as np
from PIL import Image
import os
import sys
import cv2
import logging
from datetime import datetime
from scipy.io import savemat
import matplotlib as mpl
import matplotlib.pyplot as plt


caffe_root = '../../../'
sys.path.insert(0, caffe_root + 'python')
import caffe

EPSILON = 1e-8

cats = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']

colormaps = ['#000000', '#7F0000', '#007F00', '#7F7F00', '#00007F', '#7F007F', '#007F7F', '#7F7F7F', '#3F0000', '#BF0000', '#3F7F00',
                        '#BF7F00', '#3F007F', '#BF007F', '#3F7F7F', '#BF7F7F', '#003F00', '#7F3F00', '#00BF00', '#7FBF00', '#003F7F']

def colormap(index):
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', [colormaps[0], colormaps[index+1], '#FFFFFF'], 256)
    

def caffe_forward(caffemodel, deploy_file, test_lst, label_lst, root_folder, out_folder, nGPU, flip, iters):
    logging.info('Beginning caffe_forward...')
    caffe.set_mode_gpu()
    caffe.set_device(nGPU)
    caffe.SGDSolver.display = 0
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    line = 'There are totally {} test images'.format(len(test_lst))
    logging.info(line)
    #test_scale = [241, 321, 401, 481, 561]
    test_size = 224
    for i in range(len(test_lst)):
        im_name = '{}/JPEGImages/{}.jpg'.format(root_folder, test_lst[i])
        im_labels = label_lst[i]
        #im_name = './demo_image/obs/2008_003762.jpg'
        #im_labels = [14]
        
        img = cv2.imread(im_name)
        img = np.array(img, dtype=np.float32)
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	height, width = img.shape[:2]
	min_size = min(height, width)
        scale = float(test_size) / float(min_size)
        img = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        #img = cv2.resize(img, (256, 256))
        img -= np.array((104.00698793,116.66876762,122.67891434))
        img = img.transpose((2, 0, 1))
	for label in im_labels:
            net.blobs['data'].reshape(1, *img.shape)
            net.blobs['data'].data[...] = img
            net.blobs['label'].reshape(1, 1, 1, 20)
            net.blobs['label'].data[0,0,0,label] = 1
            net.forward()
            #prob1 = net.blobs['prob'].data[0,label]
            att1 = net.blobs['score_b1'].data[0][label]
            att1[att1 < 0] = 0
	    max_value1 = np.max(att1)
	    min_value1 = np.min(att1)
            att1 = (att1 - min_value1) / (max_value1 - min_value1 + EPSILON)
          
            out = np.array(att1 * 255, dtype=np.uint8)
	    #out = cv2.resize(out, (width, height), interpolation=cv2.INTER_CUBIC)

            #out = im_gray * 0.2 + out * 0.7
            out_name = '{}/{}_{}.png'.format(out_folder, test_lst[i], label)
            #plt.imsave(out_name, out, cmap=colormap(label))
            cv2.imwrite(out_name, out)
	    #cv2.imwrite(out_name1, out)

        if (i + 1) % 50 == 0:
            line = 'Processed {} images'.format(i+1)
            print line

def load_dataset(test_lst):
    logging.info('Beginning loading dataset...')
    im_lst = []
    label_lst = []
    with open(test_lst) as f:
        test_names = f.readlines()
    lines = open(test_lst).read().splitlines()
    for line in lines:
        fields = line.split()
        im_name = fields[0]
        im_labels = []
        for i in range(len(fields)-1):
            im_labels.append(int(fields[i+1]))
        im_lst.append(im_name)
        label_lst.append(im_labels)
    return im_lst, label_lst

if __name__ == '__main__':
    iters = int(sys.argv[1])
    nGPU = int(sys.argv[2])

    caffemodel = './snapshot/memory1a_iter_{}.caffemodel'.format(iters)
    deploy_file = './deploy_integral.prototxt'

    root_folder = '../../../data/VOCdevkit/VOC2012'
    save_folder = './'
    train_lst = '../../../data/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt'
    im_lst, label_lst = load_dataset(train_lst)
    
    out_folder = save_folder + 'atten_model/'

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    #N = 2000
    N = len(im_lst)
    caffe_forward(caffemodel, deploy_file, im_lst[:N], label_lst[:N], root_folder, out_folder, nGPU, False, iters)
