import mxnet as mx

from config import cfg
from symbol.enet_symbol import get_loss, make_block,level1,initila_block
from symbol.rnn_unit import sequence2sequence


def level2(data, name, bn_momentum):

    num_filter = 128
    data = make_block(name="bottleneck2.0" + name, data=data, num_filter=num_filter, bn_momentum=bn_momentum, down_sample=True, up_sample=False)
    data = make_block(name="bottleneck2.1" + name, data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    return data

def level3_block(data, name, bn_momentum):

    num_filter = 256
    data = make_block(name="bottleneck3.0" + name, data=data, num_filter=num_filter, bn_momentum=bn_momentum, down_sample=True, up_sample=False)
    data = make_block(name="bottleneck3.1" + name, data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    return data

def level3(data, left2right, bn_momentum):

    num_filter = 256
    data0 = data = make_block(name="bottleneck3.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=True)
    data = mx.sym.Concat(data,left2right)
    num_filter = 512
    data1 = data = make_block(name="bottleneck3.1", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=True)
    num_filter = 1024
    data = make_block(name="bottleneck3.2", data=data, num_filter=num_filter,
                      bn_momentum=bn_momentum, down_sample=True)

    num_filter = 512
    data = make_block(name="bottleneck3.3", data=data, num_filter=num_filter,
                      bn_momentum=bn_momentum, up_sample=True)
    data = data + data1
    num_filter = 256
    data = make_block(name="bottleneck3.4", data=data, num_filter=num_filter,
                      bn_momentum=bn_momentum, up_sample=True)
    data = data0 + data
    data = make_block(name="bottleneck3.5", data=data, num_filter=128, bn_momentum=bn_momentum, dilated=(1, 1),up_sample=True)
    return data

def rnn(bn_momentum,height,width,is_sparse=False, net_type = 'stereo'):

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    if net_type == 'stereo':
        output_channel = 1
    elif net_type == 'flow':
        output_channel = 2

    label1 = mx.sym.Variable(net_type + '_downsample1')

    init_img1 = initila_block(img1, 'img1')
    init_img2 = initila_block(img2, 'img2')

    level1_img1 = level1(init_img1, 'img1', bn_momentum, 4)
    level1_img2 = level1(init_img2, 'img2', bn_momentum, 4)

    level2_img1 = level2(level1_img1, 'img1', bn_momentum)
    level2_img2 = level2(level1_img2, 'img2', bn_momentum)

    level3_img1 = level3_block(level2_img1, 'img1', bn_momentum)
    level3_img2 = level3_block(level2_img2, 'img2', bn_momentum)

    level3_left2right = sequence2sequence(level3_img1, level3_img2, height/16, width/16, True, cfg.RNN.num_hidden)
    data = mx.sym.Concat(level1_img1, level1_img2)

    #level 2
    num_filter = 128
    data0 = data = make_block(name="bottleneck2.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=True, up_sample=False)

    data = mx.sym.Concat(data, level2_img1, level2_img2)
    data = make_block(name="bottleneck2.1", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(2, 2))
    data = make_block(name="bottleneck2.2", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = data0 + data
    data = make_block(name="bottleneck2.3", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(4, 4))
    data = make_block(name="bottleneck2.4", data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    data2 = level3(data, level3_left2right, bn_momentum)
    data = mx.sym.Concat(data, data2)

    num_filter = 128
    data0 = data = make_block(name="bottleneck2.5", data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    data = make_block(name="bottleneck2.6", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(2, 2))
    data = make_block(name="bottleneck2.7", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5)
    data = data + data0
    data = make_block(name="bottleneck2.8", data=data, num_filter=num_filter, bn_momentum=bn_momentum)

    ## level 4
    num_filter = 64
    data = make_block(name="bottleneck4.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum,
                              up_sample=True)
    data0 = data = make_block(name="bottleneck4.1", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)
    data = mx.sym.Concat(data, level1_img1, level1_img2)
    data = make_block(name="bottleneck4.2", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)
    data = data + data0

    ## level 5
    num_filter = 32
    data  = make_block(name="bottleneck5.0", data=data, num_filter=num_filter,
                       bn_momentum=bn_momentum, up_sample=True)
    data0 = mx.sym.Concat(data, init_img1, init_img1)
    data = make_block(name="bottleneck5.1", data=data0, num_filter=num_filter,  bn_momentum=bn_momentum)

    data1 = mx.symbol.Convolution(name='conv', data=data0, num_filter=output_channel, kernel=(5, 5),
                                 stride=(1, 1), pad=(2, 2), dilate=(1, 1), no_bias=False)
    data = mx.sym.Concat(data, data1)
    data = mx.symbol.Convolution(name='pr', data=data, num_filter=output_channel, kernel=(3, 3),
                                 stride=(1, 1), pad=(1, 1), dilate=(1, 1), no_bias=False)
    loss = get_loss(data,label1, 1.0, name='loss1')

    return loss