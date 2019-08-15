import cv2
import sys
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from PIL import Image
import os
import json
THRESHOLD = 0.
class AttentionStoreProbAVELayer(caffe.Layer):
    def setup(self, bottom, top):
        ##config 
        params = eval(self.param_str)
        self.memory_dir = params['memory_dir']
        self.num_imgs = 10582
        self.stats = {}
        self.idx = 0
        self.total = 30000 * 5
        self.threshold = THRESHOLD
        self.attention_dir = params['atten_dir']
        self.probs = {}
        if not os.path.exists(self.memory_dir):
            os.mkdir(self.memory_dir)
        if not os.path.exists(self.attention_dir):
            os.mkdir(self.attention_dir)

        if len(bottom) != 5:
            raise exception('need to define three bottoms')
        if len(top) != 1:
            raise exception('only need to define one top')
    
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data
        self.idx += 1
        #if self.idx < self.num_imgs * 2:
        #    return
        self.mask = bottom[0].data[0].copy()
        label = bottom[1].data.reshape(20)
        num_cls = len(np.where(label > 0)[0])
        image_idx = int(bottom[2].data.reshape(1))
        flip = int(bottom[3].data.reshape(1))
        prob = bottom[4].data.reshape(20).tolist()
        sorted_prob = sorted(prob, reverse=True)
        
        self.mask[self.mask < 0] = 0
        if flip == 1:
            self.mask = self.mask[:,:,::-1]
        
        for i in range(20):
            if int(label[i]) == 1:
            #####normalize attention map
                ma = np.max(self.mask[i])
                mi = np.min(self.mask[i])
                self.mask[i] = (self.mask[i] - mi) / (ma - mi + 1e-8)         
                image_cls_idx = str(image_idx) + '_' + str(i) 
                cur_mem_name = self.memory_dir + image_cls_idx + '.png'
                #######read memory map
                if os.path.exists(cur_mem_name):
                    mem_im = cv2.imread(cur_mem_name, 0) / 255.
                    if sorted_prob.index(prob[i]) < num_cls:
                        attention_name = self.attention_dir + image_cls_idx + '_b1_' + str(self.stats[image_cls_idx]+1) + '.png'
                        cv2.imwrite(attention_name, self.mask[i]*255)
                        ####average 
                        mem_im1 = (mem_im * self.stats[image_cls_idx] + self.mask[i]) / (self.stats[image_cls_idx] + 1)
                        mem_im1 = np.array(mem_im1 * 255, dtype=np.uint8)
                        self.stats[image_cls_idx] += 1
                        cv2.imwrite(cur_mem_name, mem_im1)
                        
                #######save memory map    
                else:
                    if sorted_prob.index(prob[i]) < num_cls:
                        self.stats[image_cls_idx] = 1
                        mem_im = np.array(self.mask[i] * 255, dtype=np.uint8)
                        cv2.imwrite(cur_mem_name, mem_im)
                        
                        attention_name = self.attention_dir + image_cls_idx + '_b1_' + str(self.stats[image_cls_idx]) + '.png'
                        cv2.imwrite(attention_name, mem_im)
        
        
    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff


class AttentionStoreProbMAXLayer(caffe.Layer):
    def setup(self, bottom, top):
        ##config 
        params = eval(self.param_str)
        self.memory_dir = params['memory_dir']
        self.num_imgs = 10582
        self.stats = {}
        self.idx = 0
        self.total = 30000 * 5
        self.threshold = THRESHOLD
        self.attention_dir = params['atten_dir']
        self.probs = {}
        if not os.path.exists(self.memory_dir):
            os.mkdir(self.memory_dir)
        if not os.path.exists(self.attention_dir):
            os.mkdir(self.attention_dir)

        if len(bottom) != 5:
            raise exception('need to define three bottoms')
        if len(top) != 1:
            raise exception('only need to define one top')
    
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data
        self.idx += 1
        if self.idx < self.num_imgs * 1:
            return
        self.mask = bottom[0].data[0].copy()
        label = bottom[1].data.reshape(20)
        num_cls = len(np.where(label > 0)[0])
        image_idx = int(bottom[2].data.reshape(1))
        flip = int(bottom[3].data.reshape(1))
        prob = bottom[4].data.reshape(20).tolist()
        sorted_prob = sorted(prob, reverse=True)
        
        self.mask[self.mask < 0] = 0
        if flip == 1:
            self.mask = self.mask[:,:,::-1]
        
        for i in range(20):
            if int(label[i]) == 1:
            #####normalize attention map
                ma = np.max(self.mask[i])
                mi = np.min(self.mask[i])
                self.mask[i] = (self.mask[i] - mi) / (ma - mi + 1e-8)         
                image_cls_idx = str(image_idx) + '_' + str(i) 
                cur_mem_name = self.memory_dir + image_cls_idx + '.png'
                #######read memory map
                if os.path.exists(cur_mem_name):
                    mem_im = cv2.imread(cur_mem_name, 0) / 255.
                    if sorted_prob.index(prob[i]) < num_cls:
                        attention_name = self.attention_dir + image_cls_idx + '_b1_' + str(self.stats[image_cls_idx]+1) + '.png'
                        cv2.imwrite(attention_name, self.mask[i]*255)
                        #####max
                        mi = np.min(mem_im)
                        ma = np.max(mem_im)
                        mem_im = (mem_im - mi) / (ma - mi + 1e-8) 
                        mem_im1 = np.maximum(mem_im, self.mask[i]) 
                        mem_im1 = np.array(mem_im1 * 255, dtype=np.uint8)
                        self.stats[image_cls_idx] += 1
                        cv2.imwrite(cur_mem_name, mem_im1)
                        
                #######save memory map    
                else:
                    if sorted_prob.index(prob[i]) < num_cls:
                        self.stats[image_cls_idx] = 1
                        mem_im = np.array(self.mask[i] * 255, dtype=np.uint8)
                        cv2.imwrite(cur_mem_name, mem_im)
                        
                        attention_name = self.attention_dir + image_cls_idx + '_b1_' + str(self.stats[image_cls_idx]) + '.png'
                        cv2.imwrite(attention_name, mem_im)
        
        
    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff

class AttentionStoreProbHARMLayer(caffe.Layer):
    def setup(self, bottom, top):
        ##config 
        params = eval(self.param_str)
        self.memory_dir = params['memory_dir']
        self.num_imgs = 10582
        self.stats = {}
        self.idx = 0
        self.total = 30000 * 5
        self.threshold = THRESHOLD
        self.th1 = 0.2
        self.th2 = 0.
        self.attention_dir = params['atten_dir']
        self.probs = {}
        if not os.path.exists(self.memory_dir):
            os.mkdir(self.memory_dir)
        if not os.path.exists(self.attention_dir):
            os.mkdir(self.attention_dir)

        if len(bottom) != 5:
            raise exception('need to define three bottoms')
        if len(top) != 1:
            raise exception('only need to define one top')
    
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data
        self.idx += 1
        #if self.idx < self.num_imgs * 2:
        #    return
        self.mask = bottom[0].data[0].copy()
        label = bottom[1].data.reshape(20)
        num_cls = len(np.where(label > 0)[0])
        image_idx = int(bottom[2].data.reshape(1))
        flip = int(bottom[3].data.reshape(1))
        prob = bottom[4].data.reshape(20).tolist()
        sorted_prob = sorted(prob, reverse=True)
        
        self.mask[self.mask < 0] = 0
        if flip == 1:
            self.mask = self.mask[:,:,::-1]
        
        for i in range(20):
            if int(label[i]) == 1:
            #####normalize attention map
                ma = np.max(self.mask[i])
                mi = np.min(self.mask[i])
                self.mask[i] = (self.mask[i] - mi) / (ma - mi + 1e-8)         
                image_cls_idx = str(image_idx) + '_' + str(i) 
                cur_mem_name = self.memory_dir + image_cls_idx + '.png'
                #######read memory map
                if os.path.exists(cur_mem_name):
                    mem_im = cv2.imread(cur_mem_name, 0) / 255.
                    if sorted_prob.index(prob[i]) < num_cls:
                        attention_name = self.attention_dir + image_cls_idx + '_b1_' + str(self.stats[image_cls_idx]+1) + '.png'
                        cv2.imwrite(attention_name, self.mask[i]*255)
                        ####average 
                        mem_im1 = (mem_im * self.stats[image_cls_idx] + self.mask[i]) / (self.stats[image_cls_idx] + 1)
                        mi1 = np.min(mem_im1)
                        ma1 = np.max(mem_im1)
                        mem_im1 = (mem_im1 - mi1) / (ma1 - mi1 + 1e-8) 
                        #####max
                        mi2 = np.min(mem_im)
                        ma2 = np.max(mem_im)
                        mem_im = (mem_im - mi2) / (ma2 - mi2 + 1e-8) 
                        mem_im2 = np.maximum(mem_im, self.mask[i]) 
                        #### weights ave
                        mem_im3 = 3.0 / (1.0 / (mem_im1 + 1e-8) + 2.0 / (mem_im2 + 1e-8)) 
                        self.mask[i] = mem_im3
                        
                        mem_im3 = np.array(mem_im3 * 255, dtype=np.uint8)
                        self.stats[image_cls_idx] += 1
                        cv2.imwrite(cur_mem_name, mem_im3)
                        
                #######save memory map    
                else:
                    if sorted_prob.index(prob[i]) < num_cls:
                        self.stats[image_cls_idx] = 1
                        mem_im = np.array(self.mask[i] * 255, dtype=np.uint8)
                        cv2.imwrite(cur_mem_name, mem_im)
                        
                        attention_name = self.attention_dir + image_cls_idx + '_b1_' + str(self.stats[image_cls_idx]) + '.png'
                        cv2.imwrite(attention_name, mem_im)
        
        
    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff
