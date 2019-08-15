import caffe
import numpy as np


class SigmoidSplitpnLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # loss output is scalar
        top[0].reshape(1)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, bottom, top):
        
        input_data = bottom[0].data[0].copy()
        target = bottom[1].data[0].copy()
        
        self.pos = np.where(target > 0)   
        self.pos_data = input_data[self.pos]
        self.pos_label = target[self.pos]
        self.neg = np.where(target == 0)   
        self.neg_data = input_data[self.neg]
        
        pos_loss = -np.log(self.sigmoid(self.pos_data))
        neg_loss = -np.log(np.exp(-np.maximum(self.neg_data, 0)) + 1e-8) + np.log(1 + np.exp(-np.abs(self.neg_data)))
        
        loss = 0. 
        if len(self.pos_data) > 0:
            loss += 1.0 / len(self.pos_data) * np.sum(pos_loss)
        if len(self.neg_data) > 0:
            loss += 1.0 / len(self.neg_data) * np.sum(neg_loss)

        top[0].data[...] = loss 

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            if len(self.pos_data) > 0:
                bottom[0].diff[0, ...][self.pos] = -1.0 / len(self.pos_data) * (1 - self.sigmoid(self.pos_data))
            if len(self.neg_data) > 0:
                bottom[0].diff[0, ...][self.neg] = 1.0 / len(self.neg_data) * self.sigmoid(self.neg_data)


class SigmoidSplitpnProbLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # loss output is scalar
        top[0].reshape(1)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, bottom, top):
        
        input_data = bottom[0].data[0].copy()
        target = bottom[1].data[0].copy()
        
        self.pos = np.where(target > 0)   
        self.pos_data = input_data[self.pos]
        self.pos_label = target[self.pos]
        self.neg = np.where(target == 0)   
        self.neg_data = input_data[self.neg]
        
        pos_loss = -self.pos_label * np.log(self.sigmoid(self.pos_data))
        neg_loss = -np.log(np.exp(-np.maximum(self.neg_data, 0)) + 1e-8) + np.log(1 + np.exp(-np.abs(self.neg_data)))
        
        loss = 0. 
        if len(self.pos_data) > 0:
            loss += 1.0 / len(self.pos_data) * np.sum(pos_loss)
        if len(self.neg_data) > 0:
            loss += 1.0 / len(self.neg_data) * np.sum(neg_loss)

        top[0].data[...] = loss 

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            if len(self.pos_data) > 0:
                bottom[0].diff[0, ...][self.pos] = -1.0 / len(self.pos_data) * self.pos_label * (1 - self.sigmoid(self.pos_data))
            if len(self.neg_data) > 0:
                bottom[0].diff[0, ...][self.neg] = 1.0 / len(self.neg_data) * self.sigmoid(self.neg_data)


class SigmoidSplitpositiveProbLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # loss output is scalar
        top[0].reshape(1)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, bottom, top):
        self.input_data = bottom[0].data[0].copy()
        self.target = bottom[1].data[0].copy()
        pos_loss = -self.target * np.log(self.sigmoid(self.input_data))
        c, h, w = self.input_data.shape
        self.N = c * h * w
        top[0].data[...] = 1.0 /  self.N * np.sum(pos_loss) 

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[0, ...] = -1.0 / self.N * self.target * (1 - self.sigmoid(self.input_data))
            
class SigmoidSplitpProbLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # loss output is scalar
        top[0].reshape(1)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, bottom, top):
        
        input_data = bottom[0].data[0].copy()
        target = bottom[1].data[0].copy()
        
        self.pos = np.where(target > 0)   
        self.pos_data = input_data[self.pos]
        self.pos_label = target[self.pos]
        
        pos_loss = -self.pos_label * np.log(self.sigmoid(self.pos_data))
       
        loss = 0. 
        if len(self.pos_data) > 0:
            loss += 1.0 / len(self.pos_data) * np.sum(pos_loss)

        top[0].data[...] = loss 

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[0, ...] = 0.0
            if len(self.pos_data) > 0:
                bottom[0].diff[0, ...][self.pos] = -1.0 / len(self.pos_data) * self.pos_label * (1 - self.sigmoid(self.pos_data))
                
class SigmoidSplitnProbLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # loss output is scalar
        top[0].reshape(1)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, bottom, top):
        
        input_data = bottom[0].data[0].copy()
        target = bottom[1].data[0].copy()
       
        self.neg = np.where(target == 0)   
        self.neg_data = input_data[self.neg]
        neg_loss = -np.log(np.exp(-np.maximum(self.neg_data, 0)) + 1e-8) + np.log(1 + np.exp(-np.abs(self.neg_data)))
        
        loss = 0. 
        if len(self.neg_data) > 0:
            loss += 1.0 / len(self.neg_data) * np.sum(neg_loss)

        top[0].data[...] = loss 

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[0, ...] = 0.0
            if len(self.neg_data) > 0:
                bottom[0].diff[0, ...][self.neg] = 1.0 / len(self.neg_data) * self.sigmoid(self.neg_data)

