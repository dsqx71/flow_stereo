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

# def disp2color(disp):

def flow2color(flow, is_cv2imshow=False):
    """
    plot optical flow
    Parameters
    ----------
    flow : ndarray
      optical flow have 2 channel : u ,v indicate displacement

    """
    hsv = np.zeros(flow.shape[:2]+(3,)).astype(np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if is_cv2imshow:

        cv2.imshow('flow', rgb)
        cv2.waitKey(1)
    else:
        plt.figure()
        plt.imshow(rgb)
        plt.title('optical flow')


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

def readPFM(file):
    """
    read .PFM file
    Parameters
    ----------
    file : str
        file dir

    Returns
    -------
    data : ndarray
    scale : float
    """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)

    return data, scale


def writePFM(file, image, scale=1):
    """
    write .PFM file
    Parameters
    ----------
    file : str
        output dir
    image : ndarray
    scale : float
    """
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
    image = np.flipud(image)
    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))
    endian = image.dtype.byteorder
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale
    file.write('%f\n' % scale)
    image.tofile(file)

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

def plot_velocity_vector(flow,interval=30,is_cv2imshow=False):
    """
    use arrow line to draw optical flow
    Parameters
    ----------
    flow :  ndarray
        optical flow
    """

    img = np.ones(flow.shape[:2]+(3,))
    for i in range(0,img.shape[0]-20,interval):
        for j in range(0,img.shape[1]-20,interval):
            try:
                # opencv 3.1.0
                if flow.shape[-1] == 2:
                    cv2.arrowedLine(img,(j,i),(j+int(round(flow[i,j,0])),i+int(round(flow[i,j,1]))),(150,0,0),2)
                else:
                    cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i ), (150, 0, 0), 2)
            except AttributeError:
                # opencv 2.4.8
                if flow.shape[-1] == 2:
                    cv2.line(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))), (150, 0, 0), 2)
                else:
                    cv2.line(img,pt1 =  (j, i),pt2= (j + int(round(flow[i, j])), i), color = (150, 0, 0),thickness =  1)
    if is_cv2imshow:

        cv2.imshow('vector', img)
        cv2.waitKey(1)
    else:
        plt.figure()
        plt.imshow(img)
        plt.title('velocity vector')


def weight_median_filter(i, left, radius, epsilon, mask):
    """
    Constant Time Weighted Median Filtering for Stereo Matching and Beyond
    Parameters
    ----------
    i : ndarray
        disparity
    left : ndarray
        original image
    radius : int
    epsilon : float
    mask: ndarray of boolean
        indicate which need to be changed

    Returns
    -------
    dispout : ndarray
        filted disparity
    """

    dispin  = i.copy()
    dispout = dispin.copy()
    dispout[mask] = 0
    vecdisp = np.unique(dispin)

    tot = np.zeros(i.shape)
    imgaccum = np.zeros(i.shape)

    gf = filters.GuidedFilterColor(left.copy(), radius, epsilon)

    for d in vecdisp:
        if d<=0:
            continue
        ab = gf._computeCoefficients((dispin==d).astype(float))
        weight = gf._computeOutput(ab, gf._I)
        tot = tot + weight

    for d in vecdisp:
        if d<=0:
            continue
        ab = gf._computeCoefficients((dispin==d).astype(float))
        weight = gf._computeOutput(ab, gf._I)
        imgaccum = imgaccum + weight
        musk =  (imgaccum > 0.5*tot) & (dispout==0) & (mask) & (tot> 0.0001)
        dispout[musk] = d

    return dispout

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

def check_data(img1, img2, gt, interval=10, number=20):
    """
    check the validity of disparity ground truth
    Parameters
    ----------
    img1 : ndarray
        left image
    img2 : ndarray
        right image
    gt : ndarray
        disparity ground truth
    interval : int
        interval between adjacent plots
    number : int
        total number of plots
    """
    tot = 0
    for i in range(20, img1.shape[0]-20, interval):
        for j in range(20, img1.shape[1]-20, interval):
            if tot>number:
                break

            if len(gt.shape) == 2 :
                # stereo
                if gt[i,j]!=gt[i,j]:
                    continue
                print 'disparity : {}' .format(gt[i,j])
                plt.figure()
                plt.imshow(img1[i - 15:i + 16, j - 15:j + 16])
                # plt.waitforbuttonpress()
                plt.figure()
                plt.imshow(img2[i - 15:i + 16, j - int(round(gt[i, j])) - 15:j + 16 - int(round(gt[i, j]))])
                # plt.waitforbuttonpress()
                tot += 1
            else:
                # flow
                print 'flow x : {} y : {}'.format(gt[i,j,0], gt[i,j,1])
                plt.figure()
                plt.imshow(img1[i - 15:i + 16, j - 15:j + 16])
                # plt.waitforbuttonpress()
                plt.figure()
                plt.imshow(img2[i - 15 + gt[i, j, 1]:i + 16 + gt[i, j, 1], j + gt[i, j, 0] - 15:j + 16 + gt[i, j, 0]])
                # plt.waitforbuttonpress()
                tot += 1



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

def get_gradreq(net):
    """
    set gradiant requirement
    Parameters
    ----------
    net : symbol

    Returns
    -------
    grad_req : dict
        dict of gradient requirements
    """
    grad_req = {}
    for key in net.list_arguments():
        if 'img1' == key or 'img2'==key or 'stereo' in key or 'flow' in key:
            grad_req[key] = 'null'
        else:
            grad_req[key] = 'write'
    return grad_req

class Adam(mx.optimizer.Optimizer):
    """Adam optimizer as described in [King2014]_.

    .. [King2014] Diederik Kingma, Jimmy Ba,
       *Adam: A Method for Stochastic Optimization*,
       http://arxiv.org/abs/1412.6980

    the code in this class was adapted from
    https://github.com/mila-udem/blocks/blob/master/blocks/algorithms/__init__.py#L765

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.002.
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
        Default value is set to 0.9.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
        Default value is set to 0.999.
    epsilon : float, optional
        Default value is set to 1e-8.
    decay_factor : float, optional
        Default value is set to 1 - 1e-8.

    wd : float, optional
        L2 regularization coefficient add to all the weights
    rescale_grad : float, optional
        rescaling factor of gradient.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 decay_factor=(1 - 1e-8), **kwargs):
        super(Adam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.count = 0
        self.decay_factor = decay_factor

    def create_state(self, index, weight):
        """Create additional optimizer state: mean, variance

        Parameters
        ----------
        weight : NDArray
            The weight data

        """
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        lr = self._get_lr(index)
        self._update_count(index)

        t = self._index_update_count[index]
        mean, variance = state

        grad *= self.rescale_grad
        if self.clip_gradient is not None:
            clip(grad, -self.clip_gradient, self.clip_gradient, out=grad)

        mean *= self.beta1
        mean += grad * (1. - self.beta1)

        variance *= self.beta2
        variance += (1 - self.beta2) * square(grad, out=grad)

        coef1 = 1. - self.beta1 ** t
        coef2 = 1. - self.beta2 ** t
        lr *= math.sqrt(coef2) / coef1

        weight -= lr * mean / (sqrt(variance) + self.epsilon)

        wd = self._get_wd(index)
        if wd > 0.:
            weight[:] -= (lr * wd) * weight
        self.count+=1
        # if self.count>50:
        # if  np.isnan(weight.asnumpy()).any() == True:
        #     print self.idx2name[index]
        # print self.idx2name[index]
        # print grad.asnumpy().mean()
        # print weight.asnumpy().mean()