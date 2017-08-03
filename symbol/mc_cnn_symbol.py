import mxnet as mx
import numpy as np
var_registrar = {}
def get_variable(name, shape=None, init=None):
    global var_registrar
    if name not in var_registrar:
        var_registrar[name] = mx.sym.Variable(name, shape=shape, init=init, dtype=np.float32)
    return var_registrar[name]

def bn(name, data, momentum=0.95, eps = 1e-5 + 1e-10):

    beta = get_variable(name=name+ '_bn_beta')
    gamma = get_variable(name=name + 'bn_gamma')
    moving_mean= get_variable(name=name + 'bn_mean')
    moving_var= get_variable(name=name + 'bn_var')

    return  mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=momentum, eps=eps, name=name + '_bn',
                             beta=beta, gamma=gamma, moving_mean = moving_mean, moving_var = moving_var)

def get_conv(name, data, num_filter, kernel, stride, pad, with_relu, with_bn=False, dilate=(1, 1), is_conv=True):

    weight = get_variable(name=name+'_weight')
    bias = get_variable(name=name+'_bias')
    gamma = get_variable(name=name+'_gamma')
    if is_conv:
        conv = mx.symbol.Convolution(name=name, data=data, num_filter=num_filter, kernel=kernel,
                                     weight=weight, bias=bias,
                                     stride=stride, pad=pad, dilate=dilate, no_bias=False)
    else:
        conv = mx.sym.Deconvolution(name=name, data=data, num_filter=num_filter, kernel=kernel,
                                    weight=weight, bias=bias,
                                    stride=stride, pad=pad, dilate=dilate, no_bias=False)
    if with_bn:
        conv = bn(name=name, data=conv)
    return (mx.sym.LeakyReLU(data = conv,  act_type = 'prelu', gamma=gamma) if with_relu else conv)

def conv_share(sym, name, with_bn):

    data = get_conv(name='conv0_' + name, data=sym,  num_filter= 64, kernel=(5, 5), stride=(1, 1), pad=(0, 0), dilate=(1, 1), with_relu=True, with_bn=with_bn)
    data = get_conv(name='conv1_' + name, data=data, num_filter= 128, kernel=(3, 3), stride=(1, 1), pad=(0, 0), dilate=(1, 1), with_relu=True, with_bn=with_bn)
    data = get_conv(name='conv2_' + name, data=data, num_filter= 128, kernel=(3, 3), stride=(1, 1), pad=(0, 0), dilate=(1, 1), with_relu=True, with_bn=with_bn)
    data = get_conv(name='conv3_' + name, data=data, num_filter= 128, kernel=(3, 3), stride=(1, 1), pad=(0, 0), dilate=(1, 1), with_relu=True, with_bn=with_bn)
    data = get_conv(name='conv4_' + name, data=data, num_filter= 128, kernel=(3, 3), stride=(1, 1), pad=(0, 0), dilate=(1, 1), with_relu=True, with_bn=with_bn)

    return data

def get_network(network_type, with_bn=False):
    """
    Yann LeCun et al "Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches"
    """
    img1 = mx.sym.Variable('left')
    img2 = mx.sym.Variable('right')
    img1d = mx.sym.Variable('left_downsample')
    img2d = mx.sym.Variable('right_downsample')
    label = mx.sym.Variable('label')

    embedding1 = conv_share(img1, 'origin', with_bn)
    embedding2 = conv_share(img2, 'origin', with_bn)

    embedding1d = conv_share(img1d, 'downsample', with_bn)
    embedding2d = conv_share(img2d, 'downsample', with_bn)

    if network_type!='fully':
        data0 = mx.sym.Reshape(data=embedding1, shape=(-1, 1, 128))
        data1 = mx.sym.Reshape(data=embedding2, shape=(-1, 128, 1))
        data2 = mx.sym.Reshape(data=embedding1d, shape=(-1, 1, 128))
        data3 = mx.sym.Reshape(data=embedding2d, shape=(-1, 128, 1))

        s1 = mx.sym.batch_dot(data0, data1)
        s2 = mx.sym.batch_dot(data2, data3)

        s1 = mx.sym.Reshape(data=s1, shape=(-1, 1, 1, 1))
        s2 = mx.sym.Reshape(data=s2, shape=(-1, 1, 1, 1))

        c1 = mx.sym.Convolution(data=s1, no_bias=True, kernel=(1, 1), num_filter=1, name='w1')
        c2 = mx.sym.Convolution(data=s2, no_bias=True, kernel=(1, 1), num_filter=1, name='w2')

        c1 = mx.sym.Flatten(c1)
        c2 = mx.sym.Flatten(c2)
        net = c1 + c2
        net = mx.sym.LinearRegressionOutput(data=net, label=label)

        return net
    else:
        net  = mx.sym.Group([embedding1, embedding2, embedding1d, embedding2d])
        return net