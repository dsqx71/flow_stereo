import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import cv2
import sys
import re
import mxnet.ndarray as nd
from config import cfg
import math
# from guided_filter.core import filters
from mxnet.ndarray import NDArray, zeros, clip, sqrt
import os
import pandas as pd
import subprocess

def flow2color(flow):
    """
        plot optical flow
        optical flow have 2 channel : u ,v indicate displacement
    """
    hsv = np.zeros(flow.shape[:2]+(3,)).astype(np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    plt.figure()
    plt.imshow(rgb)
    plt.title('optical flow')

def estimate_label_size(net, batch_shape):
    """
        estimate label shape given by input shape
    """
    args = dict(zip(net.list_outputs(), net.infer_shape(img1=batch_shape, img2=batch_shape)[1]))
    shapes = []
    for key in args:
        shapes.append(args[key][2:])
    return shapes


def load_checkpoint(prefix, epoch, net, batchsize):

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

def load_model(name, epoch, net, batch_shape, ctx, grad_req):

    data_sym = ['img1', 'img2']
    args,aux = load_checkpoint(name, epoch)
    executor = net.simple_bind(ctx=ctx, grad_req=grad_req, img1=batch_shape, img2=batch_shape,
                               label = (1,1,375,1242))

    for key in executor.arg_dict:
        if key in data_sym or 'stereo' in key or 'flow' in key:
            executor.arg_dict[key][:] = mx.nd.zeros((executor.arg_dict[key].shape), ctx)
        else:
            if key in args:
                executor.arg_dict[key][:] = args[key]

    for key in executor.aux_dict:
        executor.aux_dict[key][:] = aux[key]

    return executor

def readPFM(file):
    """
        read .PFM file
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
    '''
       residual > 3  and   residual / gt > 0.05   (defined by kitti)
    '''

    outlier = np.zeros(gt.shape)
    mask = gt > 0

    gt = np.round(gt[mask])
    pred = pred[mask]
    err = np.abs(pred-gt)
    outlier[mask] = err

    return (err[err>tau]/(gt[err>tau].astype(np.float32)+1) > 0.05).sum()/float(mask.sum()),outlier

def plot_velocity_vector(flow):
    '''
        use arrow line to draw optical flow
    '''
    img = np.ones(flow.shape[:2]+(3,))
    for i in range(0,img.shape[0]-20,30):
        for j in range(0,img.shape[1]-20,30):
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
    plt.figure()
    plt.imshow(img)
    plt.title('velocity vector')


def weight_median_filter(i, left, radius, epsilon, mask):
    '''
        Constant Time Weighted Median Filtering for Stereo Matching and Beyond
    '''
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
    '''
        generate image record
    '''
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

def check_data(img1,img2,gt):
    """
        check the validity of disparity
    """

    tot = 0
    for i in range(100,img1.shape[0]-50,10):
        for j in range(100,img1.shape[1]-50,10):
            if gt[i,j] >0 and j-gt[i,j]>=25:
                if tot>20:
                    break
                print gt[i,j]
                plt.figure()
                plt.imshow(img1[i-15:i+16,j-15:j+16])
                plt.waitforbuttonpress()
                plt.figure()
                plt.imshow(img2[i-15:i+16,j-gt[i,j]-15:j+16-gt[i,j]])
                plt.waitforbuttonpress()
                tot += 1

class Adam(mx.optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 decay_factor=(1 - 1e-8), num_ctx=1, **kwargs):
        super(Adam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.num_ctx = num_ctx

    def create_state(self, index, weight):

        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):

        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        lr = self._get_lr(index)
        self._update_count(index)

        t = self._index_update_count[index]
        mean, variance = state

        grad *= self.rescale_grad
        if self.clip_gradient is not None:
            clip(grad, -self.clip_gradient, self.clip_gradient, out=grad)

        mean[:] = self.beta1 * mean + (1. - self.beta1) * grad
        variance[:] = self.beta2 * variance + (1. - self.beta2) * grad * grad

        coef1 = 1. - self.beta1**t
        coef2 = 1. - self.beta2**t
        lr *= math.sqrt(coef2)/coef1

        weight[:] -= lr*mean/(sqrt(variance) + self.epsilon)

        wd = self._get_wd(index)
        if wd > 0.:
            weight[:] -= (lr * wd) * weight
        # if 'left2right' in self.idx2name[index]:
        #     print self.idx2name[index]
        #     print weight.asnumpy().mean()
        #     print grad.asnumpy().mean()
        grad[:] = zeros(grad.shape, grad.context, dtype=grad.dtype)

def get_idx2name(net):
    """
        get symbol name from index
    """
    idx2name = {}
    arg_name = net.list_arguments()
    param_name = [key for key in arg_name if key !='img1' and key!='img2' and 'stereo' not in key and 'flow' not in key]
    for i,name in enumerate(param_name):
        idx2name[i] = name
    return idx2name

def get_gradreq(net):
    """
        get gradiant requirement
    """
    grad_req = {}
    for key in net.list_arguments():
        if 'img1' == key or 'img2'==key or 'stereo' in key or 'flow' in key:
            grad_req[key] = 'null'
        else:
            grad_req[key] = 'write'
    return grad_req

