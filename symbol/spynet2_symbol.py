import mxnet as mx
from symbol.dispnet_symbol import get_loss
from symbol.res_unit import residual_unit
from config import cfg

def conv_unit(sym, name, weights, bias):

    # conv0
    conv0 = get_conv(name='conv0.0' + name, data=sym, num_filter=13, kernel=(5, 5), stride=(1, 1), pad=(2, 2),
                     with_relu=True, bn_momentum=0.9, dilate=(1, 1), weight=weights[0], bias=bias[0])
    conv0 = mx.sym.Concat(conv0, sym)

    for i in range(1, 3):
        conv0 = get_conv(name='conv0.{}'.format(i) + name, data=conv0, num_filter=32, kernel=(3, 3), stride=(1,1), pad=(1,1),
             with_relu=True, bn_momentum=0.9, dilate=(1, 1), weight=weights[i], bias=bias[i])

    # conv1
    tmp = conv1 = get_conv(name='conv1.0' + name, data=conv0, num_filter=64, kernel=(5, 5), stride=(2, 2), pad=(2, 2),
                     with_relu=True, bn_momentum=0.9, dilate=(1, 1), weight=weights[3], bias=bias[3])

    for i in range(1, 3):
        conv1 = get_conv(name='conv1.{}'.format(i) + name, data=conv1, num_filter=64, kernel=(3, 3), stride=(1, 1),
                         pad=(1, 1), with_relu=True, bn_momentum=0.9, dilate=(1, 1), weight=weights[i+3], bias=bias[i+3])
    conv1 = tmp + conv1

    # conv2
    tmp = conv2 = get_conv(name='conv2.0' + name, data=conv1, num_filter=128, kernel=(5, 5), stride=(2, 2), pad=(2, 2),
                           with_relu=True, bn_momentum=0.9, dilate=(1, 1), weight=weights[6], bias=bias[6])

    for i in range(1, 3):
        conv2 = get_conv(name='conv2.{}'.format(i) + name, data=conv2, num_filter=128, kernel=(3, 3), stride=(1, 1),
                         pad=(1, 1), with_relu=True, bn_momentum=0.9, dilate=(1, 1), weight=weights[i + 6],
                         bias=bias[i + 6])
    conv2 = tmp + conv2

    return conv0,conv1,conv2

def upsample(prev_feature, prev_pr, encoder_feature, num_filter, name):

    if prev_feature is not None:
        upconv = mx.sym.Deconvolution(prev_feature, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=num_filter,
                                      name='upconv' + name, no_bias=False)
        upconv = mx.sym.LeakyReLU(data=upconv, act_type='leaky', slope=0.1)

    uppr = mx.sym.Deconvolution(prev_pr, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=1,
                                name='upsample' + name, no_bias=False)
    concat = mx.sym.Concat(encoder_feature, upconv, uppr) if prev_feature is not None else mx.sym.Concat(encoder_feature, uppr)
    feature =get_conv(name='iconv' + name, data=concat, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                         with_relu=True, bn_momentum=0.9)
    return feature, uppr


def stereo_net(net_type = 'stereo', is_sparse = False,
               loss0_scale=cfg.MODEL.loss0_scale, loss1_scale=cfg.MODEL.loss1_scale, loss2_scale=cfg.MODEL.loss2_scale,
               loss3_scale=cfg.MODEL.loss3_scale, loss4_scale=cfg.MODEL.loss4_scale, loss5_scale=cfg.MODEL.loss5_scale):


    output_dim = 1

    downsample0 = mx.sym.Variable(net_type + '_downsample1')
    downsample1 = mx.sym.Variable(net_type + '_downsample2')
    downsample2 = mx.sym.Variable(net_type + '_downsample3')
    downsample3 = mx.sym.Variable(net_type + '_downsample4')
    downsample4 = mx.sym.Variable(net_type + '_downsample5')
    downsample5 = mx.sym.Variable(net_type + '_downsample6')

    weights = [mx.sym.Variable('share{}_weight'.format(i)) for i in range(100)]
    bias    = [mx.sym.Variable('share{}_bias'.format(i)) for i in range(100)]

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    data = mx.sym.Concat(img1, img2)

    conv1 = residual_unit(data=data, num_filter=64, stride=(2, 2), dim_match=False, name='conv1',
                          bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)
    conv2 = residual_unit(data=conv1, num_filter=128, stride=(2, 2), dim_match=False, name='conv2',
                          bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)

    conv0_img1, conv1_img1, conv2_img1 = conv_unit(img1, 'img1', weights, bias)
    conv0_img2, conv1_img2, conv2_img2 = conv_unit(img2, 'img2', weights, bias)

    corr0 = mx.sym.Correlation1D(data1=conv0_img1, data2=conv0_img2, pad_size=2, kernel_size=1,
                                max_displacement=2, stride1=1, stride2=1, single_side=-1)
    corr1 = mx.sym.Correlation1D(data1=conv1_img1, data2=conv1_img2, pad_size=8, kernel_size=1,
                                max_displacement=8, stride1=1, stride2=1, single_side=-1)
    corr2 = mx.sym.Correlation1D(data1=conv2_img1, data2=conv2_img2, pad_size=64, kernel_size=1,
                                max_displacement=64, stride1=1, stride2=1, single_side=-1)

    concat = mx.sym.Concat(corr2, conv2)
    conv3 = residual_unit(data=concat, num_filter=256, stride=(2, 2), dim_match=False, name='conv3',
                           bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)
    conv4 = residual_unit(data=conv3, num_filter=512, stride=(2, 2), dim_match=False, name='conv4.0',
                           bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)
    conv4 = residual_unit(data=conv4, num_filter=512, stride=(1, 1), dim_match=False, name='conv4.1',
                          bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)
    conv5 = residual_unit(data=conv4, num_filter=512, stride=(2, 2), dim_match=False, name='conv5.0',
                           bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)
    conv5 = residual_unit(data=conv5, num_filter=512, stride=(1, 1), dim_match=False, name='conv5.1',
                          bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)

    pr5 = mx.sym.Convolution(conv5, pad=(2, 2), kernel=(5, 5), stride=(1, 1), num_filter=output_dim, name='pr5')

    iconv4, uppr5 = upsample(prev_feature=None, prev_pr=pr5, encoder_feature=conv4, num_filter=256,name='v4')
    pr4 = mx.sym.Convolution(iconv4, pad=(2, 2), kernel=(5, 5), stride=(1, 1), num_filter=output_dim, name='pr4')
    pr4 = pr4 + uppr5

    iconv3, uppr4 = upsample(prev_feature=iconv4, prev_pr=pr4, encoder_feature=conv3, num_filter=128, name='v3')
    pr3 = mx.sym.Convolution(iconv3, pad=(2, 2), kernel=(5, 5), stride=(1, 1), num_filter=output_dim, name='pr3')
    pr3 = pr3 + uppr4

    iconv2, uppr3 = upsample(prev_feature=iconv3, prev_pr=pr3, encoder_feature=concat, num_filter=64, name='v2')
    pr2 = mx.sym.Convolution(iconv2, pad=(2, 2), kernel=(5, 5), stride=(1, 1), num_filter=output_dim, name='pr2')
    pr2 = pr2 + uppr3

    concat = mx.sym.Concat(conv1, corr1)
    iconv1, uppr2 = upsample(prev_feature=iconv2, prev_pr=pr2, encoder_feature=concat, num_filter=32, name='v1')
    pr1 = mx.sym.Convolution(iconv1, pad=(2, 2), kernel=(5, 5), stride=(1, 1), num_filter=output_dim, name='pr1')
    pr1 = pr1 + uppr2

    concat = mx.sym.Concat(img1, img2, corr0)
    iconv0, uppr1 = upsample(prev_feature=iconv1, prev_pr=pr1, encoder_feature=concat, num_filter=32, name='v0')
    pr0 = mx.sym.Convolution(iconv0, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr0')
    pr0 = pr0 + uppr1

    loss0 = get_loss(pr0, downsample0, loss0_scale, name='loss0', get_data=False, is_sparse=is_sparse, type=net_type)
    loss1 = get_loss(pr1, downsample1, loss1_scale, name='loss1', get_data=False, is_sparse=is_sparse, type=net_type)
    loss2 = get_loss(pr2, downsample2, loss2_scale, name='loss2', get_data=False, is_sparse=is_sparse, type=net_type)
    loss3 = get_loss(pr3, downsample3, loss3_scale, name='loss3', get_data=False, is_sparse=is_sparse, type=net_type)
    loss4 = get_loss(pr4, downsample4, loss4_scale, name='loss4', get_data=False, is_sparse=is_sparse, type=net_type)
    loss5 = get_loss(pr5, downsample5, loss5_scale, name='loss5', get_data=False, is_sparse=is_sparse, type=net_type)

    net = mx.sym.Group([loss0, loss1, loss2, loss3, loss4, loss5])

    return net


