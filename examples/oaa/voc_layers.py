import sys
caffe_root = '../../'
sys.path.insert(0, caffe_root+'python')
import caffe
import cv2
import numpy as np
from PIL import Image
import os 

import random

class VOCDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        # config
        self.voc_dir = '../../data/VOCdevkit/VOC2012/JPEGImages/'
        self.split = 'train'
        self.mean = np.array((104.007, 116.669, 122.679))
        self.random = True
        self.seed = random.randint(0, 65536)
        self.short_size = 224
        self.crop_size = 224
        self.flip = random.randint(0, 1)
        self.num = 0
        self.num_imgs = 10582
        # two tops: data and label
        if len(top) != 4:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f = '../../data/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt'
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        #if 'train' not in self.split:
        #    self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx].split()[0])
        classlabel = np.zeros((1, 20), dtype = np.uint8)
        for i in range(len(self.indices[self.idx].split()) - 1):
            classlabel[0, int(self.indices[self.idx].split()[i+1])] = 1 
        self.label = classlabel
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        #top[1].reshape(1, 1, *self.label.shape)
        top[1].reshape(*self.label.shape)
        top[2].reshape(1, 1, 1, 1)
        top[3].reshape(1, 1, 1, 1)


    def forward(self, bottom, top):
        # assign output
        self.num += 1
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.idx
        top[3].data[...] = self.flip
    
        self.flip = random.randint(0, 1)
        
        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/{}.jpg'.format(self.voc_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        height, width = in_.shape[:2]
        min_size = min(height, width)
        scale = float(self.short_size) / float(min_size)
        in_ = cv2.resize(in_, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        #in_ = cv2.resize(in_, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
        #rh = random.randint(0, in_.shape[0] - self.crop_size - 1)
        #rw = random.randint(0, in_.shape[1] - self.crop_size - 1)
        #in_ = in_[rh:rh+self.crop_size, rw:rw+self.crop_size, :]

        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))

        if self.flip == 1:
            in_ = in_[:,:,::-1]
        return in_

class VOCData1Layer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        # config
        self.voc_dir = '../../data/VOCdevkit/VOC2012/JPEGImages/'
        self.att_dir = './exp1/memory1ar'
        self.att_model_dir = './exp1/atten_model/'
        self.split = 'train'
        self.mean = np.array((104.007, 116.669, 122.679))
        self.random = True
        self.seed = random.randint(0, 65536)
        self.short_size = 224
        self.crop_size = 224
        self.flip = random.randint(0, 1)
        self.num = 0
        self.num_imgs = 10582
        # two tops: data and label
        if len(top) != 5:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f = '../../data/VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt'
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        #if 'train' not in self.split:
        #    self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx].split()[0])
        self.attention = self.load_attention(self.indices[self.idx].split()[0])
        classlabel = np.zeros((1, 20), dtype = np.uint8)
        for i in range(len(self.indices[self.idx].split()) - 1):
            classlabel[0, int(self.indices[self.idx].split()[i+1])] = 1 
        self.label = classlabel
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        #top[1].reshape(1, 1, *self.label.shape)
        top[1].reshape(*self.label.shape)
        top[2].reshape(1, 1, 1, 1)
        top[3].reshape(1, 1, 1, 1)
	top[4].reshape(1, *self.attention.shape)

    def forward(self, bottom, top):
        # assign output
        self.num += 1
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.idx
        top[3].data[...] = self.flip
        top[4].data[...] = self.attention    		
        self.flip = random.randint(0, 1)
        
        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass

    def load_attention(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        tmp_name = '{}/{}_{}.png'.format(self.att_dir, idx, self.indices[self.idx].split()[1])
        if not os.path.exists(tmp_name):
            print 'hahah'
            tmp_name = '{}/{}_{}.png'.format(self.att_model_dir, idx, self.indices[self.idx].split()[1])
        tmp = Image.open(tmp_name)
        tmp = np.array(tmp, dtype=np.float32)
        height, width = tmp.shape 
        atts = np.zeros((20, height, width), dtype=np.float32)
        for i in range(len(self.indices[self.idx].split()) - 1):
            la = int(self.indices[self.idx].split()[i+1]) 
            att_name = '{}/{}_{}.png'.format(self.att_dir, idx, la)
            if not os.path.exists(att_name):            
                att_name = '{}/{}_{}.png'.format(self.att_model_dir, idx, la)
            att = Image.open(att_name)
            att_ = np.array(att, dtype=np.float32) / 255.0

            if self.flip == 1:
		att_  = att_[:, ::-1]
            atts[la, :, :] = att_
        
        return atts
 
    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/{}.jpg'.format(self.voc_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        height, width = in_.shape[:2]
        min_size = min(height, width)
        scale = float(self.short_size) / float(min_size)
        in_ = cv2.resize(in_, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        #in_ = cv2.resize(in_, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
        #rh = random.randint(0, in_.shape[0] - self.crop_size - 1)
        #rw = random.randint(0, in_.shape[1] - self.crop_size - 1)
        #in_ = in_[rh:rh+self.crop_size, rw:rw+self.crop_size, :]

        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))

        if self.flip == 1:
            in_ = in_[:,:,::-1]
        return in_
        
