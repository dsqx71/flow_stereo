#-*- coding:utf-8 -*-
from skimage import io
import mxnet as mx
from random import shuffle,randint,randrange
from model import get_network
import matplotlib.pyplot as plt
import numpy as np
import cv2

def init(key,weight,value):
    if 'bias' in key:
        weight[:] = 0
    if key == 'w1_weight' or key == 'w2_weight':
        weight[:] = mx.random.uniform(0,value,weight.shape) 
    else:
        weight[:] = mx.random.uniform(-value,value,weight.shape) 

def output_embedding(img_dir,epoch,ctx):  
    '''
        fully embedding
    '''
    left  = io.imread(img_dir[1])
    right = io.imread(img_dir[2])
    left_downsample  = cv2.resize(left, (0,0), fx=0.5, fy=0.5)
    right_downsample = cv2.resize(right,(0,0), fx=0.5, fy=0.5)
    
    left =( left - left.reshape(-1,3).mean(axis=0) )/left.reshape(-1,3).std(axis=0)
    right=( right- right.reshape(-1,3).mean(axis=0))/right.reshape(-1,3).std(axis=0)

    left_downsample = ( left_downsample - left_downsample.reshape(-1,3).mean(axis=0) )/left_downsample.reshape(-1,3).std(axis=0)
    right_downsample=( right_downsample- right_downsample.reshape(-1,3).mean(axis=0))/right_downsample.reshape(-1,3).std(axis=0)
    
    left =  left.swapaxes(2,1).swapaxes(1,0) 
    right= right.swapaxes(2,1).swapaxes(1,0) 
    
    left_downsample  =  left_downsample.swapaxes(2,1).swapaxes(1,0) 
    right_downsample = right_downsample.swapaxes(2,1).swapaxes(1,0) 
    
    s1 = (1,3,left.shape[1],left.shape[2])
    s2 = (1,3,left_downsample.shape[1],left_downsample.shape[2])
    net,executor =  load_model('stereo',epoch,s1,s2,'fully',ctx,1)
    
    args  = dict(zip(net.list_arguments(),executor.arg_arrays))
    args['left'][:] = np.array([left])
    args['right'][:] = np.array([right])
    args['left_downsample'][:] = np.array([left_downsample])
    args['right_downsample'][:]= np.array([right_downsample])
    executor.forward(is_train=False)
    return args,executor.outputs[0].asnumpy()[0],executor.outputs[1].asnumpy()[0],executor.outputs[2].asnumpy()[0],executor.outputs[3].asnumpy()[0]

def get_kitty_data_dir2012(low,high):
    img_dir = []
    for num in range(low,high):
        dir_name = '000{}'.format(num)
        if len(dir_name) ==4 :
            dir_name = '00'+dir_name
        elif len(dir_name) == 5:
            dir_name = '0'+dir_name
        gt = './disp_noc/'+dir_name+'_10.png'.format(num)
        imgL = './colored_0/'+dir_name+'_10.png'.format(num)
        imgR = './colored_1/'+dir_name+'_10.png'.format(num)
        img_dir.append((gt,imgL,imgR))
    return img_dir

def get_kitty_data_dir2015(low,high):
    img_dir = []
    for num in range(low,high):
        dir_name = '000{}'.format(num)
        if len(dir_name) ==4 :
            dir_name = '00'+dir_name
        elif len(dir_name) == 5:
            dir_name = '0'+dir_name
        gt = './disp_noc_0/'+dir_name+'_10.png'.format(num)
        imgL = './image_2/'+dir_name+'_10.png'.format(num)
        imgR = './image_3/'+dir_name+'_10.png'.format(num)
        img_dir.append((gt,imgL,imgR))
    return img_dir

def assign_grad_req(net):
    grad_req = {}
    for key in net.list_arguments():
        if key =='w1_weight' or key =='w2_weight':
            grad_req[key] = 'write'
        else:
            grad_req[key] = 'add'
    return grad_req

def load_model(name,epoch,s1,s2,network_type,ctx,batch_size):

    data_sign = ['left','right','left_downsample','right_downsample','label']
    net,args,aux = mx.model.load_checkpoint(name,epoch)
    keys = net.list_arguments()
    net = get_network(network_type,batch_size)
    grad_req = assign_grad_req(net)
    executor = net.simple_bind(ctx=ctx,grad_req=grad_req,left_downsample=s2,right_downsample=s2,left = s1,right= s1)
    
    for key in executor.arg_dict:
        if key in  data_sign:
            executor.arg_dict[key][:] = mx.nd.zeros((executor.arg_dict[key].shape),ctx)
        else:
            if key in args:
                executor.arg_dict[key][:] = args[key]
            else:
                init(key,executor.arg_dict[key],0.01)
    return net,executor

def draw_patch(args,executor,img_idx):
    fig = plt.figure()
    plt.xticks(visible=False) 
    for i in range(4):
        p1 = fig.add_subplot(4,4,1+i*4)
        p2 = fig.add_subplot(4,4,2+i*4)
        p3 = fig.add_subplot(4,4,3+i*4)
        p4 = fig.add_subplot(4,4,4+i*4)
        l_p =  args['left'].asnumpy()[i].swapaxes(0,1).swapaxes(1,2) + 128
        r_p = args['right'].asnumpy()[i].swapaxes(0,1).swapaxes(1,2) + 128
        ld  = args['left_downsample'].asnumpy()[i].swapaxes(0,1).swapaxes(1,2) + 128
        rd  = args['right_downsample'].asnumpy()[i].swapaxes(0,1).swapaxes(1,2) + 128
        result = (executor.outputs[0].asnumpy()[i],args['label'].asnumpy()[i])
        
        p1.imshow(l_p)
        p2.imshow(r_p)
        p3.imshow(ld)
        p4.imshow(rd)    
        plt.title('gt: %d score: %.5f ' % (result[1],result[0]))
    plt.savefig('./result/img_%d_gt_%d_matchingscore_%.5f.jpg' % (img_idx,result[1],result[0]))
    plt.close()   

def produce_stereo_matching(dirs,ctx,dis_range,epoch_num):
    args1,l,r,ld,rd = output_embedding(dirs,epoch_num,ctx)
    _,args2,_ = mx.model.load_checkpoint('stereo',epoch_num)
    w1 = args2['w1_weight'].asnumpy()[0][0][0][0]
    w2 = args2['w2_weight'].asnumpy()[0][0][0][0]

    ld = cv2.resize(ld.swapaxes(0,1).swapaxes(1,2),(l.shape[2],r.shape[1])).swapaxes(2,1).swapaxes(1,0)
    rd = cv2.resize(rd.swapaxes(0,1).swapaxes(1,2),(l.shape[2],r.shape[1])).swapaxes(2,1).swapaxes(1,0)
    
    ms_left  = np.zeros((l.shape[1],l.shape[2],dis_range))
    ms1_left = np.zeros((l.shape[1],l.shape[2],dis_range))
    ms2_left = np.zeros((l.shape[1],l.shape[2],dis_range))
    dis_left = np.zeros((l.shape[1],l.shape[2]))
    
    ms_right  = np.zeros((l.shape[1],l.shape[2],dis_range))
    ms1_right = np.zeros((l.shape[1],l.shape[2],dis_range))
    ms2_right = np.zeros((l.shape[1],l.shape[2],dis_range))
    dis_right = np.zeros((l.shape[1],l.shape[2]))

    for y in range(l.shape[1]):
        l_ms  =  mx.nd.array(l[:,y].T,ctx)
        r_ms  =  mx.nd.array(r[:,y]  ,ctx)
        ld_ms = mx.nd.array(ld[:,y].T,ctx)
        rd_ms = mx.nd.array(rd[:,y],ctx)

        tmp1 = mx.nd.dot(l_ms,r_ms)
        tmp2 = mx.nd.dot(ld_ms,rd_ms)
        tmp = w1*tmp1 + w2*tmp2
        tmp1 = tmp1.asnumpy()
        tmp2 = tmp2.asnumpy()
        tmp  = tmp.asnumpy()
        
        for x in range(l.shape[2]):
            if x - (dis_range-1) <0:
                t1 = 0
            else:
                t1 = x - dis_range + 1 
                
            if x + dis_range  > l.shape[2]:
                t2 = l.shape[2]
            else:
                t2 =  x + dis_range
            length1 = len(tmp[x,t1:x+1])
            length2 = len(tmp[x:t2,x])
      
            dis_right[y,x] =  tmp[x:t2,x].argmax()
            dis_left[y,x]  =  tmp[x,t1:x+1][::-1].argmax()
          
            ms_left[y,x,:length1]  =  tmp[x,t1:x+1][::-1]
            ms_right[y,x,:length2] =  tmp[x:t2,x]
            
            if length1!=dis_range:
                ms_left[y,x,length1:] = np.ones(dis_range-length1) * 0.3
            if length2!=dis_range:
                ms_right[y,x,length2:]= np.ones(dis_range-length2) * 0.3
    return dis_left,dis_right,ms_left,ms_right,w1,w2

def outlier_sum(pred,gt,tau=3):
    
    outlier = np.zeros(gt.shape)
    mask = gt > 0
    gt = np.round(gt[mask]/256.0)
    pred = pred[mask]
    err = np.abs(pred-gt)
    outlier[mask] = err
    plt.figure()
    plt.imshow(outlier)
    plt.close()
    return (err[err>tau]/gt[err>tau].astype(np.float32) > 0.05).sum()/float(mask.sum()),outlier

def compute_unary(ms,class_num,GT_PROB,tau):
    '''
       MRF unary energy
    '''
    u = ms.reshape(-1,class_num).T
    u[0,:] = 0.0
    tot = u.T.sum(axis=1)
    u = u/tot
    mask = u.max(axis=0) < tau
    u[:,mask] = np.ones((class_num,sum(mask))) * 0.00001
    u = np.ascontiguousarray(u)
    return (-np.log(u)).astype(np.float32)