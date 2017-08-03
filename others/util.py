import mxnet.ndarray as nd
import cv2
import os
from ..data import config

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

def outlier_sum(pred, label, tau=3):
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
    # mask of valid points
    mask = (label == label)

    #error length
    err = label - pred
    err = cv2.pow(err, 2)
    err = cv2.pow(err.sum(axis=0), 0.5)

    #label vector length
    label_length = cv2.pow(label, 2)
    label_length = cv2.pow(label_length.sum(axis=0), 0.5)

    # num of valid points
    num_valid = float(mask[0].sum())

    err = err[mask[0]]
    label_length = label_length[mask[0]]

    # outlier =  residual > 3  and  residual / gt > 0.05
    return ( err[err>tau] / (label_length[err>tau] + 1E-3) > 0.05 ).sum() / num_valid

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

def generate_deployconfig(experiment_name, data_type, type='end2end'):
    """
    Gernerate deploy config for prediction pipeline
    """
    file_path = os.path.join(config.cfg.model.model_zoo, experiment_name+'.config')
    if type == 'end2end':
        with open(file_path, 'w') as f:
            f.write('[model]\n')
            f.write('model_type = {}\n'.format(data_type))
            f.write('model_prefix = {}\n'.format(experiment_name))
            f.write('ctx = 0\n')
            f.write('need_preprocess = 1\n')
    elif type == 'patch_match':
        with open(file_path, 'w') as f:
            f.write('[model]\n')
            f.write('model_type = {}\n'.format(data_type))
            f.write('model_prefix = {}\n'.format(experiment_name))
            f.write('ctx = 0\n')
            f.write('need_preprocess = 1\n')
            f.write('max_displacement = 227\n')
            f.write('batchsize = 128\n')

def update_arg_params(arg_params, low_level_name, high_level_name):

    """
        Initialize current level model from low level model
    """
    arg_names = arg_params.keys()
    for name in arg_names:
        if low_level_name in name:
            new_name = name.replace(low_level_name, high_level_name)
            arg_params[new_name] = arg_params[name].copy()
    return arg_params