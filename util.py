import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import cv2
import sys
import re


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

    shapes = sorted(shapes, key=lambda t: t[0], reverse=True)
    return shapes


def init_param(scale, args):
    """
        initialize parameters
    """
    init = mx.initializer.Normal(sigma=scale)
    for key in args:
        if 'img1' not in key and 'img2' not in key and 'flow' not in key and 'stereo' not in key:
            if 'bias' in key:
                args[key][:] = 0.0
            else:
                init(key,args[key])


def load_model(name, epoch, net, batch_shape, ctx, network_type='write'):
    """
        load parameter trained before and simple bind
        return executor
    """
    data_sym = ['img1', 'img2']
    _, args, _ = mx.model.load_checkpoint(name, epoch)
    executor = net.simple_bind(ctx=ctx, grad_req=network_type, img1=batch_shape, img2=batch_shape)
    init = mx.init.Orthogonal(scale=1.0, rand_type='normal')

    for key in executor.arg_dict:
        if key in data_sym or 'stereo' in key or 'flow' in key:
            executor.arg_dict[key][:] = mx.nd.zeros((executor.arg_dict[key].shape), ctx)
        else:
            if key in args:
                executor.arg_dict[key][:] = args[key]
            else:
                init(key, executor.arg_dict[key])

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