# -*- coding:utf-8 -*-  
import mxnet as mx
import numpy as np
from collections import namedtuple
from skimage import io
from sklearn import utils
from random import randint,shuffle,uniform,choice
import cv2
import matplotlib.pyplot as plt


def get_network(network_type,batch_size):
    relu = {}
    conv = {}
    weight = {}
    bias = {}
    relu[0]  = mx.sym.Variable('left')
    relu[1]  = mx.sym.Variable('right')
    relu[2]  = mx.sym.Variable('left_downsample')
    relu[3]  = mx.sym.Variable('right_downsample')
    label    = mx.sym.Variable('label')
    for num_layer in range(1,5):
        #blue parameter
        weight[0]   = mx.sym.Variable('l%d_blue' % num_layer)
        bias[0]     = mx.sym.Variable('bias%d_blue' % num_layer)
        #red  parameter 
        weight[1]   =  mx.sym.Variable('l%d_red' % num_layer)
        bias[1]      = mx.sym.Variable('bias%d_red' % num_layer)
        
        if num_layer <= 2:
            kernel = (3,3)
            num_filter = 32
        else:
            kernel = (5,5)
            num_filter = 200
        
        for j in range(4):
            if network_type=='fully' and num_layer == 1:
                conv[j]  = mx.sym.Convolution(data = relu[j] ,weight=weight[j/2],bias=bias[j/2],kernel=kernel,num_filter=num_filter,pad=(6,6))
                relu[j]  = mx.sym.Activation( data = conv[j], act_type="relu")
            else:
                conv[j]  = mx.sym.Convolution(data = relu[j] ,weight=weight[j/2],bias=bias[j/2],kernel=kernel,num_filter=num_filter)
                relu[j]  = mx.sym.Activation( data = conv[j], act_type="relu")
        
    if network_type!='fully':
        flatten = {}        
        for j in range(4):
            flatten[j] = mx.sym.Flatten(data=relu[j])

        s1 = mx.sym.Dotproduct(data1=flatten[0],data2=flatten[1])
        s2 = mx.sym.Dotproduct(data1=flatten[2],data2=flatten[3])

        s1 = mx.sym.Reshape(data=s1,target_shape = (batch_size,1,1,1))
        s2 = mx.sym.Reshape(data=s2,target_shape = (batch_size,1,1,1))
       
        c1 = mx.sym.Convolution(data=s1,no_bias=True,kernel=(1,1),num_filter=1,name='w1')
        c2 = mx.sym.Convolution(data=s2,no_bias=True,kernel=(1,1),num_filter=1,name='w2')
        
        c1 = mx.sym.Flatten(c1)
        c2 = mx.sym.Flatten(c2)
        net  = c1 + c2
        net  = mx.sym.LinearRegressionOutput(data = net , label = label )
        return net
    else:
        net  = mx.sym.Group([relu[0],relu[1],relu[2],relu[3]])
        return net

DataBatch = namedtuple('DataBatch', ['data', 'label', 'pad', 'index'])
class dataiter(mx.io.DataIter):

    def __init__(self,img_dir,batch_size,ctx,datatype,high,low,rotate_range=3,contrast_range=1.3,brightness_range=0.7):
    
        self.batch_size = batch_size
        self.reset()
        self.img_dir  = img_dir
        self.num_imgs = len(img_dir)
        self.datatype = datatype   
        self.ctx = ctx
        self.rotate_range     = rotate_range
        self.contrast_range   = contrast_range
        self.brightness_range = brightness_range
        self.high = high
        self.low  = low
        
    def produce_patch(self,ith):
        dis  = np.round(io.imread(self.img_dir[ith][0])/256.0).astype(int)
        left = io.imread(self.img_dir[ith][1])  
        right= io.imread(self.img_dir[ith][2])  
        
        left =( left - left.reshape(-1,3).mean(axis=0) )/left.reshape(-1,3).std(axis=0)
        right=( right- right.reshape(-1,3).mean(axis=0))/right.reshape(-1,3).std(axis=0)

        self.generate_patch_with_ground_truth(left,right,dis)  
        self.now_img = self.img_dir[ith][0]

   
    def generate_patch_with_ground_truth(self,left,right,dis):
        s1 = 6
        s2 = 13
        for y in xrange(s2,dis.shape[0]-s2):
            for x in xrange(s2,dis.shape[1]-s2):
                if dis[y,x]!=0:
                    if np.random.random()>0.30:
                        continue     
                    d = dis[y,x]
                    if x-d>=s2 :
                        self.data[0].append( left[y-s1:y+1+s1,x-s1:x+1+s1,:])
                        self.data[1].append(right[y-s1:y+1+s1,x-s1-d:x+1+s1-d,:])
                        self.data[2].append(cv2.resize(left[y-s2:y+s2,x-s2:x+s2,:], (0,0), fx=0.5, fy=0.5))
                        self.data[3].append(cv2.resize(right[y-s2:y+s2,x-s2-d:x+s2-d,:],(0,0), fx=0.5, fy=0.5))
                        self.data_augment()
                        self.labels.extend([1,1])
                        while True:
                            temp = [x - d + move for move in range(self.low,self.high+1) if x-d+move<dis.shape[1]-s2]
                            temp.extend([x - d - move for move in range(self.low,self.high+1) if x-d-move>=s2]) 
                            xn = np.random.choice(temp)
                            #xn = np.random.randint(s2,dis.shape[1]-s2-1)
                            if xn<dis.shape[1]-s2 and x-d != xn and xn>=s2:
                                break
                        self.data[0].append( left[y-s1:y+1+s1,    x-s1:x+1+s1,:])
                        self.data[1].append(right[y-s1:y+1+s1,  xn-s1:xn+1+s1,:])
                        self.data[2].append(cv2.resize(left[y-s2:y+s2,    x-s2:x+s2,:],(0,0),fx=0.5,fy=0.5))
                        self.data[3].append(cv2.resize(right[y-s2:y+s2, xn-s2:xn+s2,:],(0,0),fx=0.5,fy=0.5))  
                        self.data_augment()
                        self.labels.extend([0,0]) 
                        self.inventory += 4  
        #utils.shuffle([self.data[0],self.data[1],self.data[2],self.data[3],self.labels])
    def data_augment(self):

        rotate = randint(-self.rotate_range,self.rotate_range)
        rotation_matrix = cv2.getRotationMatrix2D((13/2, 13/2), rotate, 1)
        beta  = uniform(0,self.brightness_range)
        alpha = uniform(1,self.contrast_range)
        flip_type = choice([-1,0,1])
        for  i in range(4):
            tmp = cv2.warpAffine(self.data[i][-1], rotation_matrix, (13, 13))
            tmp = cv2.flip(tmp,flip_type)
            tmp = cv2.multiply(tmp,np.array([alpha]))   
            tmp = cv2.add(tmp,np.array([beta]))
            self.data[i].append(tmp)
           
    def reset(self):
    
        self.index = 0
        self.img_idx = 0
        self.inventory = 0
        self.data = [[],[],[],[]]
        self.labels = []

    def iter_next(self):
        if self.inventory < self.batch_size:
            if self.img_idx >= self.num_imgs:
                return False
            if self.datatype !='test':
                self.produce_patch(self.img_idx)
            else:
                self.produce_patch_test(self.img_idx) 
                #没写
            self.img_idx+=1
            return self.iter_next()
        else: 
            self.inventory -= self.batch_size
            return True

    def getdata(self):
        
        left  = mx.nd.array(np.asarray(self.data[0][:self.batch_size]).swapaxes(3,2).swapaxes(2,1),self.ctx)
        right = mx.nd.array(np.asarray(self.data[1][:self.batch_size]).swapaxes(3,2).swapaxes(2,1),self.ctx)
        left_downsample = mx.nd.array(np.asarray(self.data[2][:self.batch_size]).swapaxes(3,2).swapaxes(2,1),self.ctx)
        right_downsample = mx.nd.array(np.asarray(self.data[3][:self.batch_size]).swapaxes(3,2).swapaxes(2,1),self.ctx)
        del self.data[0][:self.batch_size]
        del self.data[1][:self.batch_size]
        del self.data[2][:self.batch_size]
        del self.data[3][:self.batch_size]
    
        return [left,right,left_downsample,right_downsample]
    
    def getlabel(self):
        if self.datatype !='test':
            result =  mx.nd.array(np.array(self.labels[:self.batch_size]))
            del self.labels[:self.batch_size]
            return result
        else :
            return None
    
    def getindex(self):
        return self.index
    
    def getpad(self):
        return 0
