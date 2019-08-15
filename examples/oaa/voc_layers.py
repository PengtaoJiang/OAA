import sys
caffe_root = '../../'
sys.path.insert(0, caffe_root+'python')
import caffe
import cv2
import numpy as np
from PIL import Image
import os 

import random

class VOC1sDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        # config
        self.voc_dir = '/home/andrew/Datasets/semantic/PASCAL_VOC2012/JPEGImages/'
        self.split = 'train'
        self.mean = np.array((104.007, 116.669, 122.679))
        self.random = True
        self.seed = random.randint(0, 65536)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f = '/home/andrew/Datasets/semantic/PASCAL_VOC2012/train_cls.txt'
        self.indices = open(split_f, 'r').read().splitlines()
        random.shuffle(self.indices) 
        self.indices = self.indices[:8466] 
        self.idx = 0
        self.flip = random.randint(0, 1)

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

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
        top[1].reshape(1, 1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

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
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_

class VOCCropDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        # config
        self.voc_dir = '/home/andrew/Datasets/semantic/PASCAL_VOC2012/JPEGImages/'
        self.split = 'train'
        self.mean = np.array((104.007, 116.669, 122.679))
        self.random = True
        self.seed = random.randint(0, 65536)
        self.short_size = 256
        self.crop_size = 224
        self.flip = random.randint(0, 1)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f = '/home/andrew/Datasets/semantic/PASCAL_VOC2012/train_cls.txt'
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


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

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
        #min_size = min(height, width)
        #scale = float(self.short_size) / float(min_size)
        #in_ = cv2.resize(in_, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        in_ = cv2.resize(in_, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
        #rh = random.randint(0, in_.shape[0] - self.crop_size - 1)
        #rw = random.randint(0, in_.shape[1] - self.crop_size - 1)
        #in_ = in_[rh:rh+self.crop_size, rw:rw+self.crop_size, :]

        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))

        if self.flip == 1:
            in_ = in_[:,:,::-1]
        return in_

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
        self.att_dir = '/media/miao/ptsdisk/memory2/exp8/memory8ar'
        self.att_model_dir = '/media/miao/ptsdisk/memory2/exp8/atten_model'
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
        
class SemiVOCDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        # config
        self.voc_dir = '/home/andrew/Datasets/semantic/PASCAL_VOC2012/JPEGImages/'
        self.gt_dir = '/home/andrew/Datasets/semantic/PASCAL_VOC2012/SegmentationClass/'
        self.split = 'train'
        self.mean = np.array((104.007, 116.669, 122.679))
        self.random = True
        self.seed = random.randint(0, 65536)
        self.short_size = 232
        self.crop_size = 224
        self.flip = random.randint(0, 1)

        # two tops: data and label
        if len(top) != 5:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f = '/home/andrew/Datasets/semantic/PASCAL_VOC2012/train_cls.txt'
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        semi_lst = '/home/andrew/Datasets/semantic/PASCAL_VOC2012/VOCTrain/voc_train_200.txt'
        self.semi_inds = open(semi_lst, 'r').read().splitlines()
        self.semi_idx = 0
        # make eval deterministic
        #if 'train' not in self.split:
        #    self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)
            self.semi_idx = random.randint(0, len(self.semi_inds)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data1 = self.load_image(self.indices[self.idx].split()[0])
        self.data2 = self.load_image(self.semi_inds[self.semi_idx])
        self.gt = self.load_gt(self.semi_inds[self.semi_idx])
        classlabel = np.zeros((1, 20), dtype = np.uint8)
        for i in range(len(self.indices[self.idx].split()) - 1):
            classlabel[0, int(self.indices[self.idx].split()[i+1])] = 1 
        self.label = classlabel
        #self.gt = np.repeat(self.gt, 20, axis=0)
        h, w = self.gt.shape
        self.new_gt = np.zeros((2, h, w), dtype=np.float32)
        self.new_gt[1, ...] = self.gt
        self.new_gt[0, ...] = 1 - self.gt
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data1.shape)
        top[1].reshape(1, *self.data2.shape)
        top[2].reshape(1, *self.new_gt.shape)
        #top[1].reshape(1, 1, *self.label.shape)
        top[3].reshape(*self.label.shape)
        top[4].reshape(*self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data1
        top[1].data[...] = self.data2
        top[2].data[...] = self.new_gt
        top[3].data[...] = self.label
        top[4].data[...] = 0
        self.flip = random.randint(0, 1)

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
            self.semi_idx = random.randint(0, len(self.semi_inds)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
            self.semi_idx += 1
            if self.semi_idx == len(self.semi_inds):
                self.semi_idx = 0


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
        self.rh = random.randint(0, in_.shape[0] - self.crop_size - 1)
        self.rw = random.randint(0, in_.shape[1] - self.crop_size - 1)
        in_ = in_[self.rh:self.rh+self.crop_size, self.rw:self.rw+self.crop_size, :]

        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))

        if self.flip == 1:
            in_ = in_[:,:,::-1]
        return in_

    def load_gt(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/{}.png'.format(self.gt_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        height, width = in_.shape[:2]
        min_size = min(height, width)
        scale = float(self.short_size) / float(min_size)
        in_ = cv2.resize(in_, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        #in_ = cv2.resize(in_, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        #rh = random.randint(0, in_.shape[0] - self.crop_size - 1)
        #rw = random.randint(0, in_.shape[1] - self.crop_size - 1)
        in_ = in_[self.rh:self.rh+self.crop_size, self.rw:self.rw+self.crop_size]
        #in_ = cv2.resize(in_, (14, 14), interpolation=cv2.INTER_NEAREST)

        in_[in_ > 0] = 1
        #in_ = in_[np.newaxis, ...]

        if self.flip == 1:
            in_ = in_[:,::-1]
        return in_

class Semi2VOCDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        # config
        self.voc_dir = '../../data/VOCdevkit/VOC2012/JPEGImages/'
        self.gt_dir = '../../data/VOCdevkit/VOC2012/SegmentationClass/'
        self.split = 'train'
        self.mean = np.array((104.007, 116.669, 122.679))
        self.random = True
        self.seed = random.randint(0, 65536)
        self.short_size = 232
        self.crop_size = 224
        self.flip = random.randint(0, 1)

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

        #semi_lst = '/home/andrew/Datasets/semantic/PASCAL_VOC2012/VOCTrain/voc_train_200.txt'
        #self.semi_inds = open(semi_lst, 'r').read().splitlines()
        #self.semi_idx = 0
        print "There are totally {} images".format(len(self.semi_inds))
        # make eval deterministic
        #if 'train' not in self.split:
        #    self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)
            self.semi_idx = random.randint(0, len(self.semi_inds)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data1 = self.load_image(self.indices[self.idx].split()[0])
        self.data2 = self.load_image(self.semi_inds[self.semi_idx])
        self.gt = self.load_gt(self.semi_inds[self.semi_idx])
        classlabel = np.zeros((1, 20), dtype = np.uint8)
        for i in range(len(self.indices[self.idx].split()) - 1):
            classlabel[0, int(self.indices[self.idx].split()[i+1])] = 1 
        self.label = classlabel
        #self.gt = np.repeat(self.gt, 20, axis=0)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data1.shape)
        top[1].reshape(1, *self.data2.shape)
        top[2].reshape(1, *self.gt.shape)
        #top[1].reshape(1, 1, *self.label.shape)
        top[3].reshape(*self.label.shape)
        top[4].reshape(*self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data1
        top[1].data[...] = self.data2
        top[2].data[...] = self.gt
        top[3].data[...] = self.label
        top[4].data[...] = 0
        self.flip = random.randint(0, 1)

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
            self.semi_idx = random.randint(0, len(self.semi_inds)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
            self.semi_idx += 1
            if self.semi_idx == len(self.semi_inds):
                self.semi_idx = 0


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
        self.rh = random.randint(0, in_.shape[0] - self.crop_size - 1)
        self.rw = random.randint(0, in_.shape[1] - self.crop_size - 1)
        in_ = in_[self.rh:self.rh+self.crop_size, self.rw:self.rw+self.crop_size, :]

        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))

        if self.flip == 1:
            in_ = in_[:,:,::-1]
        return in_

    def all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def load_gt(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/{}.png'.format(self.gt_dir, idx))
        in_ = np.array(im, dtype=int)
        height, width = in_.shape[:2]
        min_size = min(height, width)
        scale = float(self.short_size) / float(min_size)
        in_ = cv2.resize(in_, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        #in_ = cv2.resize(in_, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        #rh = random.randint(0, in_.shape[0] - self.crop_size - 1)
        #rw = random.randint(0, in_.shape[1] - self.crop_size - 1)
        in_ = in_[self.rh:self.rh+self.crop_size, self.rw:self.rw+self.crop_size]

        #in_ = in_[:,:,::-1]
        #in_ -= self.mean
        #in_ = in_.transpose((2,0,1))
        in_[in_ == 255] = 21
        #gt = in_[np.newaxis, ...]
        gt = np.zeros((22,) + in_.shape, dtype=int)
        gt[self.all_idx(in_, axis=0)] = 1

        ignore = np.where(gt[21,:,:] == 1)
        gt = gt[1:21,:,:]
        gt[:,ignore[0],ignore[1]] = 255

        if self.flip == 1:
            gt = gt[:,:,::-1]
        return gt
