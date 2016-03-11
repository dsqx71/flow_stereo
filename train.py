# -*- coding:utf-8 -*-  
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import logging
import os
import argparse
from model import get_network,dataiter
from util import get_kitty_data_dir2012,get_kitty_data_dir2015,load_model,draw_patch,init,assign_grad_req

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()  
    parser.add_argument('--continue',action='store',dest='con',type=int)
    parser.add_argument('--lr',action='store',dest='lr',type=float)
    parser.add_argument('--l',action='store',dest='low',type=int)
    parser.add_argument('--h',action='store',dest='high',type=int)

    cmd = parser.parse_args()
    batch_size = 5000
    s1 = (batch_size,3,13,13)
    s2 = (batch_size,3,27,27)
    ctx = mx.gpu(3) 
    data_sign = ['left','right','left_downsample','right_downsample','label']
    
    if cmd.con == -1:
        net = get_network('not fully',batch_size)
        grad_req = assign_grad_req(net)
        executor = net.simple_bind(ctx=ctx,grad_req=grad_req,left_downsample=s1,right_downsample=s1,left = s1,right= s1)
        keys  = net.list_arguments()
        grads = dict(zip(net.list_arguments(),executor.grad_arrays))
        args  = dict(zip(keys,executor.arg_arrays))
        auxs  = dict(zip(keys,executor.arg_arrays))
        logging.info("complete network architecture design")
    else:
        net,executor = load_model('stereo',cmd.con,s1,s1,'not fully',ctx,batch_size)
        keys = net.list_arguments()
        grads = dict(zip(keys,executor.grad_arrays))
        args  = dict(zip(keys,executor.arg_arrays))
        auxs  = dict(zip(keys,executor.arg_arrays))
        logging.info("load the paramaters and net")
       
    num_epoches = 10000
    dirs = get_kitty_data_dir2015(0,180)
    dirs.extend(get_kitty_data_dir2012(0,180))
    train_iter =  dataiter(dirs,batch_size,ctx,'train',cmd.high,cmd.low,3,1.3,0.7)
    states     =  {}
    #init
    opt = mx.optimizer.SGD(learning_rate=cmd.lr,momentum = 0.9,wd=0.00001,rescale_grad=(1.0/batch_size))
   
    for index,key in enumerate(keys):
        if key not in data_sign:
            states[key] = opt.create_state(index,args[key])
            if cmd.con == -1 :
                init(key,args[key],0.07)

    # train + validate 
    last_loss = 0.25
    for ith_epoche in range(num_epoches):    

        train_iter.reset()
        train_loss = 0.0
        nbatch = 0
        loss_of_100 = 0.0

        for dbatch in train_iter:
           
            args['left'][:]             = dbatch.data[0]
            args['right'][:]            = dbatch.data[1]
            args['left_downsample'][:]  = dbatch.data[2]
            args['right_downsample'][:] = dbatch.data[3]
            args['label'][:] = dbatch.label
            nbatch += 1
        
            executor.forward(is_train=True)
            
            #draw_patch(args,executor,train_iter.img_idx)
            loss = np.power(executor.outputs[0].asnumpy() - args['label'].asnumpy().reshape(-1,1),2).mean()
            train_loss  += loss
            loss_of_100 += loss
            tmp    = executor.outputs[0].asnumpy()
            pos_ms = tmp[args['label'].asnumpy()==1].mean()
            neg_ms = tmp[args['label'].asnumpy()==0].mean()
            logging.info("training: {}th pair img:{}th l2 loss:{} pos_ms:{} neg_ms:{} >:{} lr:{}".format(nbatch,train_iter.img_idx,loss,pos_ms,neg_ms,pos_ms-neg_ms,opt.lr))
       
            if nbatch % 30 == 0:
                loss_of_100 /=30.0
                print train_iter.now_img 
                logging.info("mean loss of 30 batches: {} ".format(loss_of_100))

            executor.backward()
            for index,key in enumerate(keys):
                if key not in data_sign:       
                    opt.update(index,args[key],grads[key],states[key])
                    grads[key][:] = np.zeros(grads[key].shape)

            if nbatch % 50 == 0:
                cmd.con = (cmd.con + 1) % 50
                mx.model.save_checkpoint('stereo',cmd.con,net,args,auxs)

        train_loss/=nbatch
        logging.info('training: ith_epoche :{} mean loss:{} last loss:{}'.format(ith_epoche,train_loss,last_loss))
        if train_loss>last_loss+0.0001**ith_epoche:
            opt.lr/= 0.1
        last_loss = train_loss

