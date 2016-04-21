import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import cv2


def flow2color(flow):
    """
        plot optical flow
    """

    hsv = np.zeros(flow.shape[:2]+(3,)).astype(np.uint8)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    plt.figure()
    plt.imshow(rgb)
    plt.title('optical flow')


def estimate_label_size(net, batch_shape):
    """
        estimate label shape given by input shape
    """

    args = dict(zip(net.list_outputs(),net.infer_shape(img1 = batch_shape,img2= batch_shape)[1]))
    shapes = []
    for key in args:
        shapes.append(args[key][2:])

    shapes = sorted(shapes, key=lambda t: t[0], reverse=True)
    return shapes


def init_param(scale, args):

    init = mx.initializer.Normal(sigma=scale)
    for key in args:
        if 'img1' not in key and 'img2' not in key and 'flow' not in key and 'stereo' not in key:
            if 'bias' in key:
                args[key][:] = 0.0
            else:
                init(key,args[key])

def load_model(name, epoch, net, batch_shape, ctx, network_type='write'):

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

