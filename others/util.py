import numpy as np
import mxnet.ndarray as nd
import os
import mxnet as mx
import pandas as pd
import subprocess
from ..data.config import cfg

def load_checkpoint(prefix, epoch):
    """
    Parameters
    ----------
    prefix: str
        directory prefix of checkpoint
    epoch: int
    Returns
    -------
    arg_params: mxnet.ndarray
    aux_params: mxnet.ndarray
    """
    save_dict = nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
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

    return (err[err>tau]/(gt[err>tau].astype(np.float32)+1) > 0.05).sum()/float(mask.sum()), outlier

def get_idx2name(net):
    """
    get indexs and corrpesonding symbol name

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
    param_name = [key for key in arg_name if key !='img1' and key!='img2'
                  and 'label' not in key]
    for i,name in enumerate(param_name):
        idx2name[i] = name
    return idx2name

