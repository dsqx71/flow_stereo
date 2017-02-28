import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import cv2
import sys
import re
import mxnet.ndarray as nd
# from config import cfg
# from guided_filter.core import filters
import os
import math
import mxnet as mx
from mxnet.ndarray import NDArray, zeros, clip, sqrt, square
# import  mxnet.optimizer.Optimizer
import pandas as pd
import subprocess
import matplotlib.pyplot as plt


def estimate_label_size(net, batch_shape):
    """
    estimate label shape given by input shape
    Parameters
    ----------
    net : symbol
    batch_shape :  tuple
        batch shape of input
    Returns
    -------
    shapes :  list
        list of label shapes
    """

    args = dict(zip(net.list_outputs(), net.infer_shape(img1=batch_shape, img2=batch_shape)[1]))
    shapes = []
    for key in args:
        shapes.append(args[key][2:])
    return shapes


def load_checkpoint(prefix, epoch):

    save_dict = nd.load('%s-%04d.params' % (prefix, epoch))
    # names_arg = net.list_arguments()
    # names_aux = net.list_auxiliary_states()
    #
    # shapes = net.infer_shape(img1left_data = batchsize, img1right_data=batchsize,
    #                         img2left_data = batchsize, img2right_data=batchsize)
    # args = dict(zip(names_arg, shapes[0]))
    # auxs = dict(zip(names_aux, shapes[2]))
    arg_params = {}
    aux_params = {}

    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':#and name in names_arg:
            arg_params[name] = v
        if tp == 'aux':# and name in names_aux:
            aux_params[name] = v

    # init = mx.init.Xavier(rnd_type='gaussian',factor_type='in',magnitude=4)
    # for name in names_arg:
    #     # if 'data' not in name and 'label' not in name:
    #     if name not in arg_params:
    #         arg_params[name] = nd.zeros(args[name])
    #         init(name,arg_params[name])
    # for name in names_aux:
    #     if name not in aux_params:
    #         aux_params[name] = nd.zeros(auxs[name])

    return arg_params,aux_params

def outlier_sum(pred,gt,tau=3):
    """
    residual > 3  and   residual / gt > 0.05   (defined by kitti)
    Parameters
    ----------
    pred : ndarray
        predict
    gt : ndarray
        ground truth
    tau : int
        threshold deciding whethe a point is outlier
    """
    outlier = np.zeros(gt.shape)
    mask = gt > 0

    gt = np.round(gt[mask])
    pred = pred[mask]
    err = np.abs(pred-gt)
    outlier[mask] = err

    return (err[err>tau]/(gt[err>tau].astype(np.float32)+1) > 0.05).sum()/float(mask.sum()),outlier

def get_imageRecord(dataset,batchsize,prefetch_buffer):
    """
    generate image record
    Parameters
    ----------
    dataset : dataset
        please refer to dataset.py
    batchsize : tuple
        the tuple has four elements
    prefetch_buffer : int
        total number of prefetch buffers

    Returns
    -------
    records : image record
    """
    data_type = dataset.data_type

    if data_type == 'flow':
        raise ValueError('do not support flow data')

    records = []
    for index,i in  enumerate(['img1','img2','label']):

        if not os.path.exists(cfg.record_prefix+'{}_{}_{}.rec'.format(dataset.name(),data_type,i)):

            df = pd.DataFrame(dataset.dirs)
            df[index].to_csv(cfg.record_prefix + '{}_{}_{}.lst'.format(dataset.name(),data_type,i),sep='\t',header=False)
            args = ['python','im2rec.py',cfg.record_prefix +'{}_{}_{}'.format(dataset.name(),data_type,i) ,\
                        '--root','','--resize','0','--quality','0','--num_thread','1','--encoding', '.png']
            print ' '.join(args)
            subprocess.call(args)

        records.append(mx.io.ImageRecordIter(
                      path_imgrec = cfg.record_prefix+'{}_{}_{}.rec'.format(dataset.name(),data_type,i),
                      data_shape = (3,) + dataset.shapes(),
                      batch_size = batchsize,
                      preprocess_threads = 1,
                      prefetch_buffer = prefetch_buffer,
                      shuffle = False))
    return records



def get_idx2name(net):
    """
    get symbol name from index
    Parameters
    ----------
    net: symbol
    Returns
    -------
    idx2name : dict
        map index to name of symbol
    """
    idx2name = {}
    arg_name = net.list_arguments()
    param_name = [key for key in arg_name if key !='img1' and key!='img2' and 'stereo' not in key and 'flow' not in key]
    for i,name in enumerate(param_name):
        idx2name[i] = name
    return idx2name

