import mxnet as mx
import numpy as np

from symbol.dispnet_symbol import SparseRegressionLoss


@mx.operator.register("argmax")
class Argmaxop(mx.operator.NumpyOp):

    def __init__(self):

        super(Argmaxop, self).__init__(True)

    def list_arguments(self):

        return ['data']

    def list_outputs(self):

        return ['output']

    def infer_shape(self, in_shape):

        data_shape = in_shape[0]
        output_shape = [in_shape[0][0],1] + in_shape[0][2:]

        return [data_shape], [output_shape]

    def forward(self,in_data,out_data):

        x = in_data[0]
        y = out_data[0]
        y[:] = np.expand_dims(np.argmax(x,axis=1),1)

    def backward(self, out_grad, in_data, out_data, in_grad):

        shape = in_data[0].shape
        y = out_data[0].ravel().astype(np.int)

        out_grad = out_grad[0]
        dx = in_grad[0]
        dx = dx.transpose(0,2,3,1).reshape(-1, shape[1])
        dx[:] = np.zeros(dx.shape)

        dx[np.arange(y.shape[0]), y] = out_grad.ravel()
        dx = dx.reshape(shape[0],shape[2],shape[3],shape[1]).transpose(0,3,1,2)
        in_grad[0][:] = dx

def get_conv(name, data, num_filter, kernel, stride, pad,
             with_relu, bn_momentum=0.9, dilate=(1, 1)):
    conv = mx.symbol.Convolution(
        name=name,
        data=data,
        num_filter=num_filter,
        kernel=kernel,
        stride=stride,
        pad=pad,
        dilate=dilate,
        no_bias=True,
        workspace= 4096
    )
    bn = mx.symbol.BatchNorm(
        name=name + '_bn',
        data=conv,
        fix_gamma=False,
        momentum=bn_momentum,
        # Same with https://github.com/soumith/cudnn.torch/blob/master/BatchNormalization.lua
        eps=1e-5 + 1e-10 # issue of cudnn
    )
    return (
        # It's better to remove ReLU here
        # https://github.com/gcr/torch-residual-networks
        mx.symbol.LeakyReLU(name=name + '_prelu', act_type='prelu', data=bn)
        if with_relu else bn
    )

def get_loss(data,label,grad_scale,name,get_data=False, is_sparse = False):

    if  is_sparse:
        loss = SparseRegressionLoss(is_l1=False, loss_scale=grad_scale)
        loss = loss(data=data, label=label)
    else:
        loss = mx.sym.MAERegressionOutput(data=data, label=label, name=name, grad_scale=grad_scale)

    return (loss,data) if get_data else loss


def initila_block(data, name):
    # TODO: input shape: (1, 3, 1086, 2173) shape incorrect
    conv = mx.symbol.Convolution(
        name="initial_conv" + name,
        data=data,
        num_filter=13,
        kernel=(3, 3),
        stride=(2, 2),
        pad=(1, 1),
        no_bias=True
    )

    maxpool = mx.symbol.Pooling(data=data, pool_type="max", kernel=(2, 2), stride=(2, 2),
                                name="initial_maxpool" + name)
    concat = mx.symbol.Concat(
        conv,
        maxpool,
        num_args=2,
        name="initial_concat" + name
    )
    return concat


def make_block(name, data, num_filter, bn_momentum,
               down_sample=False, up_sample=False,
               dilated=(1, 1), asymmetric=0):
    """maxpooling & padding"""
    if down_sample:
        # 1x1 conv ensures that channel equal to main branch
        maxpool = get_conv(name=name + '_proj_maxpool',
                           data=data,
                           num_filter=num_filter,
                           kernel=(2, 2),
                           pad=(0, 0),
                           with_relu=True,
                           bn_momentum=bn_momentum,
                           stride=(2, 2))

    elif up_sample:
        # maxunpooling.
        maxpool = mx.symbol.Deconvolution(name=name + '_unpooling',
                                   data=data,
                                   num_filter=num_filter,
                                   kernel=(4, 4),
                                   stride=(2, 2),
                                   pad=(1, 1))

        # Reference: https://github.com/e-lab/ENet-training/blob/master/train/models/decoder.lua
        # Padding is replaced by 1x1 convolution
        maxpool = get_conv(name=name + '_padding',
                           data=maxpool,
                           num_filter=num_filter,
                           kernel=(1, 1),
                           stride=(1, 1),
                           pad=(0, 0),
                           bn_momentum=bn_momentum,
                           with_relu=False)
    # main branch begin
    proj = get_conv(name=name + '_proj0',
                    data=data,
                    num_filter=num_filter,
                    kernel=(1, 1) if not down_sample else (2, 2),
                    stride=(1, 1) if not down_sample else (2, 2),
                    pad=(0, 0),
                    with_relu=True,
                    bn_momentum=bn_momentum)

    if up_sample:
        conv = mx.symbol.Deconvolution(name=name + '_deconv',
                                   data=proj,
                                   num_filter=num_filter,
                                   kernel=(4, 4),
                                   stride=(2, 2),
                                   pad=(1, 1))
    else:
        if asymmetric == 0:
            conv = get_conv(name=name + '_conv',
                            data=proj,
                            num_filter=num_filter,
                            kernel=(3, 3),
                            pad=dilated,
                            dilate=dilated,
                            stride=(1, 1),
                            with_relu=True,
                            bn_momentum=bn_momentum)
        else:
            conv = get_conv(name=name + '_conv1',
                            data=proj,
                            num_filter=num_filter,
                            kernel=(1, asymmetric),
                            pad=(0, asymmetric / 2),
                            stride=(1, 1),
                            dilate=dilated,
                            with_relu=True,
                            bn_momentum=bn_momentum)
            conv = get_conv(name=name + '_conv2',
                            data=conv,
                            num_filter=num_filter,
                            kernel=(asymmetric, 1),
                            pad=(asymmetric / 2, 0),
                            dilate=dilated,
                            stride=(1, 1),
                            with_relu=True,
                            bn_momentum=bn_momentum)

    regular = mx.symbol.Convolution(name=name + '_expansion',
                                        data=conv,
                                        num_filter=num_filter,
                                        kernel=(1, 1),
                                        pad=(0, 0),
                                        stride=(1, 1),
                                        no_bias=True)
    regular = mx.symbol.BatchNorm(
        name=name + '_expansion_bn',
        data=regular,
        fix_gamma=False,
        momentum=bn_momentum,
        eps=1e-5 + 1e-10 # issue of cudnn
    )
    # main branch end
    # TODO: spatial dropout

    if down_sample or up_sample:
        regular = mx.symbol.ElementWiseSum(maxpool, regular, name =  name + "_plus")
    # else:
        # regular = mx.symbol.ElementWiseSum(data, regular, name =  name + "_plus")
    regular = mx.symbol.LeakyReLU(name=name + '_expansion_prelu', act_type='prelu', data=regular)
    return regular

def level1(data,name,bn_momentum,num=4,down_sample=True):

    num_filter = 64
    data = data0 = make_block(name="bottleneck1.0" + name, data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=down_sample, up_sample=False)
    for block in range(num):
        data = make_block(name='bottleneck1.%d' % (block + 1) + name,
                          data=data, num_filter=num_filter,  bn_momentum=bn_momentum,
                          down_sample=False, up_sample=False)
    data0 = make_block(name="projection1" + name, data=data0, num_filter=num_filter, bn_momentum=bn_momentum)
    data = data + data0

    return data


def level3(data,bn_momentum):

    num_filter = 256
    data0 = data = make_block(name="bottleneck3.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=True)
    num_filter = 512
    data1 = data = make_block(name="bottleneck3.1", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=True)
    num_filter = 1024
    data = make_block(name="bottleneck3.2", data=data, num_filter=num_filter,
                      bn_momentum=bn_momentum, down_sample=True)
    data = make_block(name="bottleneck3.3", data=data, num_filter=num_filter,
                      bn_momentum=bn_momentum)

    num_filter = 512
    data = make_block(name="bottleneck3.4", data=data, num_filter=num_filter,
                      bn_momentum=bn_momentum, up_sample=True)
    data = data + data1
    num_filter = 256
    data = make_block(name="bottleneck3.5", data=data, num_filter=num_filter,
                      bn_momentum=bn_momentum, up_sample=True)
    data = data0 + data
    data = make_block(name="bottleneck3.6", data=data, num_filter=128, bn_momentum=bn_momentum, dilated=(1, 1),up_sample=True)
    return data

def get_body(bn_momentum,is_sparse=False, net_type = 'stereo'):

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    if net_type == 'stereo':
        output_channel = 1
    elif net_type == 'flow':
        output_channel = 2

    label1 = mx.sym.Variable(net_type + '_downsample1')

    init_img1 = initila_block(img1, 'img1')
    init_img2 = initila_block(img2, 'img2')

    corr1 = mx.sym.Correlation1D(data1=init_img1, data2=init_img2, pad_size=32, kernel_size=1,
                                 max_displacement=32, stride1=1, stride2=1,single_side=-1)

    level1_img1 = level1(init_img1, 'img1', bn_momentum, 10)
    level1_img2 = level1(init_img2, 'img2', bn_momentum, 10)

    corr2 = mx.sym.Correlation1D(data1=level1_img1, data2=level1_img2, pad_size=128, kernel_size=1,
                                max_displacement=128, stride1=1, stride2=1,single_side=-1)
    data = mx.sym.Concat(level1_img1, level1_img2, corr2)

    #level 2
    num_filter = 128
    data0 = data = make_block(name="bottleneck2.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=True, up_sample=False)
    data = make_block(name="bottleneck2.1", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(2, 2))
    data = make_block(name="bottleneck2.2", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = data0 + data
    data = make_block(name="bottleneck2.3", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(4, 4))
    data = make_block(name="bottleneck2.4", data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    data2 = level3(data, bn_momentum)
    data = mx.sym.Concat(data, data2)

    num_filter = 128
    data0 = data = make_block(name="bottleneck2.5", data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    data = make_block(name="bottleneck2.6", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(2, 2))
    data = make_block(name="bottleneck2.7", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = data + data0
    data = make_block(name="bottleneck2.8", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(4, 4))
    data = make_block(name="bottleneck2.9", data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    ## level 4
    num_filter = 64
    data = make_block(name="bottleneck4.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum,
                              up_sample=True)
    data = mx.sym.Concat(data, corr2)
    data0 = data = make_block(name="bottleneck4.1", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)
    data = mx.sym.Concat(data, level1_img1, level1_img2)
    data = make_block(name="bottleneck4.2", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)
    data = data + data0

    ## level 5
    num_filter = 32
    data  = make_block(name="bottleneck5.0", data=data, num_filter=num_filter,
                       bn_momentum=bn_momentum, up_sample=True)
    data0 = mx.sym.Concat(data, corr1, init_img1, init_img1)
    data = make_block(name="bottleneck5.1", data=data0, num_filter=num_filter,  bn_momentum=bn_momentum)

    data1 = mx.symbol.Convolution(name='conv', data=data0, num_filter=output_channel, kernel=(5, 5),
                                 stride=(1, 1), pad=(2, 2), dilate=(1, 1), no_bias=False)
    data = mx.sym.Concat(data, data1)
    data = mx.symbol.Convolution(name='pr', data=data, num_filter=output_channel, kernel=(3, 3),
                                 stride=(1, 1), pad=(1, 1), dilate=(1, 1), no_bias=False)
    loss = get_loss(data,label1, 1.0, name='loss1',is_sparse=is_sparse)

    return loss


def get_body2(bn_momentum,is_sparse=False):

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    net_type = 'stereo'
    label1 = mx.sym.Variable(net_type + '_downsample1')
    # label2 = mx.sym.Variable(net_type + '_downsample1')

    init_img1 = initila_block(img1, 'img1')
    init_img2 = initila_block(img2, 'img2')

    corr1 = mx.sym.Correlation1D(data1=init_img1, data2=init_img2, pad_size=32, kernel_size=1,
                                 max_displacement=32, stride1=1, stride2=1,single_side=-1)

    level1_img1 = level1(init_img1, 'img1', bn_momentum, 8)
    level1_img2 = level1(init_img2, 'img2', bn_momentum, 8)

    corr2 = mx.sym.Correlation1D(data1=level1_img1, data2=level1_img2, pad_size=128, kernel_size=1,
                                max_displacement=128, stride1=1, stride2=1,single_side=-1)

    data = mx.sym.Concat(level1_img1,level1_img2, corr2)

    #level 2
    num_filter = 128
    data = data0 = make_block(name="bottleneck2.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=True, up_sample=False)
    data = make_block(name="bottleneck2.1", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = make_block(name="bottleneck2.2", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(2, 2))
    data = make_block(name="bottleneck2.3", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = make_block(name="bottleneck2.4", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(4, 4))
    data = make_block(name="bottleneck2.5", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = make_block(name="bottleneck2.6", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(8, 8))
    data = make_block(name="bottleneck2.7", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    num_filter = 256
    data = make_block(name="bottleneck2.8", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(16, 16))

    data = make_block(name="bottleneck2.9", data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    data = make_block(name="bottleneck2.10", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(32, 32))
    data0 = make_block(name="projection2", data=data0, num_filter=num_filter, bn_momentum=bn_momentum)
    data = data + data0

    ##level 3
    num_filter = 64
    data = make_block(name="bottleneck4.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum,
                              up_sample=True)
    data = mx.sym.Concat(data,corr2)
    data0 = data = make_block(name="bottleneck4.1", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)
    data = make_block(name="bottleneck4.2", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)
    data = make_block(name="bottleneck4.3", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = data + data0

    pr2 = get_conv(name='pr2', data=data, num_filter=5, kernel=(3, 3), stride=(1, 1), pad=(1, 1), with_relu=False,
                   bn_momentum=bn_momentum, dilate=(1, 1))
    data = mx.sym.Concat(data, pr2)
    # ##level 4
    num_filter = 16
    data  = make_block(name="bottleneck5.0", data=data, num_filter=num_filter,
                       bn_momentum=bn_momentum,
                       up_sample=True)
    data = mx.sym.Concat(data, corr1, init_img1)
    data0 = data = make_block(name="bottleneck5.1", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)
    data = make_block(name="bottleneck5.2", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = make_block(name="bottleneck5.3", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = data + data0
    data = mx.symbol.Deconvolution(data=data, kernel=(16, 16), stride=(2, 2), num_filter=1, name="fullconv")
    data = mx.symbol.Crop(*[data, img1], name="fullconv_crop")

    # loss2 = get_loss(pr2, label2,0.8,name='loss2')
    loss = get_loss(data,label1, 1.0, name='loss1')

    return loss



def get_body3(bn_momentum,is_sparse=False, net_type='stereo'):

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    if net_type == 'stereo':
        output_channel = 1
    elif net_type == 'flow':
        output_channel = 2

    label1 = mx.sym.Variable(net_type + '_downsample1')

    init_img1 = initila_block(img1, 'img1')
    init_img2 = initila_block(img2, 'img2')

    level1_img1 = level1(init_img1, 'img1', bn_momentum, 8, down_sample=False)
    level1_img2 = level1(init_img2, 'img2', bn_momentum, 8, down_sample=False)

    corr = mx.sym.Correlation1D(data1=level1_img1, data2=level1_img2, pad_size=256, kernel_size=1,
                                max_displacement=256, stride1=1, stride2=2,single_side=-1)
    # argmax = Argmaxop()
    # corr = argmax(data=corr)
    data = mx.sym.Concat(level1_img1, level1_img2, corr)

    num_filter = 64
    data = make_block(name="bottleneck0.0", data=data, num_filter=num_filter,
                      bn_momentum=bn_momentum,
                      up_sample=False)
    data0 = data = make_block(name="bottleneck0.1", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)
    data = mx.sym.Concat(data, init_img1, init_img2)
    data = make_block(name="bottleneck0.2", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)
    data = make_block(name="bottleneck0.3", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = data + data0
    data = mx.symbol.Convolution(name='pr', data=data, num_filter=output_channel, kernel=(3,3),
                                 stride=(1,1), pad=(1,1), dilate=(1,1), no_bias=False)

    loss = get_loss(data,label1, 1.0, name='loss1')

    return loss


def get_body4(bn_momentum,is_sparse=False, net_type = 'stereo'):

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    if net_type == 'stereo':
        output_channel = 1
    elif net_type == 'flow':
        output_channel = 2

    label1 = mx.sym.Variable(net_type + '_downsample1')
    # label2 = mx.sym.Variable(net_type + '_downsample1')

    init_img1 = initila_block(img1, 'img1')
    init_img2 = initila_block(img2, 'img2')

    corr1 = mx.sym.Correlation1D(data1=init_img1, data2=init_img2, pad_size=32, kernel_size=1,
                                 max_displacement=32, stride1=1, stride2=1,single_side=-1)

    level1_img1 = level1(init_img1, 'img1', bn_momentum, 10)
    level1_img2 = level1(init_img2, 'img2', bn_momentum, 10)

    corr2 = mx.sym.Correlation1D(data1=level1_img1, data2=level1_img2, pad_size=128, kernel_size=1,
                                max_displacement=128, stride1=1, stride2=1,single_side=-1)

    data = mx.sym.Concat(level1_img1, level1_img2, corr2)

    #level 2
    num_filter = 128
    data = data0 = make_block(name="bottleneck2.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=True, up_sample=False)
    data = make_block(name="bottleneck2.1", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = make_block(name="bottleneck2.2", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(2, 2))
    data = make_block(name="bottleneck2.3", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = data0 + data
    data = make_block(name="bottleneck2.4", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(4, 4))
    data = make_block(name="bottleneck2.5", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = make_block(name="bottleneck2.6", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(8, 8))
    data = make_block(name="bottleneck2.7", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)

    data = make_block(name="bottleneck2.8", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(16, 16))

    data = make_block(name="bottleneck2.9", data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    data = make_block(name="bottleneck2.10", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(32, 32))
    data0 = make_block(name="projection2", data=data0, num_filter=num_filter, bn_momentum=bn_momentum)
    data = data + data0

    ##level 3
    num_filter = 256
    data = data0 = make_block(name="bottleneck3.1", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum)
    data = make_block(name="bottleneck3.2", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(2, 2))
    data = make_block(name="bottleneck3.3", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = make_block(name="bottleneck3.4", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(4, 4))
    num_filter = 512
    data = make_block(name="bottleneck3.5", data=data, num_filter=num_filter, bn_momentum=bn_momentum)
    data = make_block(name="bottleneck3.6", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(8, 8))
    data = make_block(name="bottleneck3.7", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = make_block(name="bottleneck3.8", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(16, 16))
    data = make_block(name="bottleneck3.9", data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    data = make_block(name="bottleneck3.10", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(32, 32))
    data0 = make_block(name="projection3", data=data0, num_filter=num_filter, bn_momentum=bn_momentum)
    data = data + data0

    num_filter = 64
    data = make_block(name="bottleneck4.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum,
                              up_sample=True)
    data = mx.sym.Concat(data, corr2)
    data0 = data = make_block(name="bottleneck4.1", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)
    data = mx.sym.Concat(data, level1_img1, level1_img2)
    data = make_block(name="bottleneck4.2", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)
    data = data + data0

    ##level 4
    num_filter = 32
    data  = make_block(name="bottleneck5.0", data=data, num_filter=num_filter,
                       bn_momentum=bn_momentum, up_sample=True)
    data = mx.sym.Concat(data, corr1, init_img1, init_img1)
    data = make_block(name="bottleneck5.1", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)
    data = mx.symbol.Convolution(name='pr', data=data, num_filter=output_channel, kernel=(3, 3),
                                 stride=(1, 1), pad=(1, 1), dilate=(1, 1), no_bias=False)

    loss = get_loss(data,label1, 1.0, name='loss1')

    return loss

