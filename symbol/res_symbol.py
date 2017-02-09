import mxnet as mx
from symbol.dispnet_symbol import get_loss
from symbol.res_unit import residual_unit
from symbol.drr_symbol import  detect_replace_refine
from symbol.enet_symbol import  get_conv
from config import cfg

def conv_unit(sym, name, weights, bias):

    conv1 = mx.sym.Convolution(data=sym,pad=(3, 3), kernel=(7, 7),stride=(2, 2),num_filter=64,
                               weight=weights[0], bias=bias[0], name='conv1' + name)
    conv1 = mx.sym.LeakyReLU(data = conv1,  act_type = 'leaky', slope  = 0.1, name='relu1'+name)

    conv2 = mx.sym.Convolution(data = conv1, pad  = (2,2),  kernel=(5,5),stride=(2,2),num_filter=128,
                                 weight = weights[1], bias = bias[1], name='conv2' + name)
    conv2 = mx.sym.LeakyReLU(data = conv2, act_type = 'leaky', slope = 0.1, name='relu2'+name)

    return conv1,conv2

def stereo_net(is_sparse = False,
               loss1_scale=0.35, loss2_scale=0.25,
               loss3_scale=0.15, loss4_scale=0.10,
               loss5_scale=0.05, loss6_scale=0.01):
    print is_sparse
    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')
    net_type = 'stereo'
    output_dim = 1
    downsample0 = mx.sym.Variable(net_type + '_downsample1')
    downsample1 = mx.sym.Variable(net_type + '_downsample2')
    downsample2 = mx.sym.Variable(net_type + '_downsample3')
    downsample3 = mx.sym.Variable(net_type + '_downsample4')
    downsample4 = mx.sym.Variable(net_type + '_downsample5')
    downsample5 = mx.sym.Variable(net_type + '_downsample6')
    downsample6 = mx.sym.Variable(net_type + '_downsample7')

    weights = [mx.sym.Variable('share{}_weight'.format(i)) for i in range(1,4)]
    bias    = [mx.sym.Variable('share{}_bias'.format(i)) for i in range(1,4)]

    data = mx.sym.Concat(img1, img2)

    conv0 = residual_unit(data=data, num_filter=32, stride=(1, 1), dim_match=False, name='conv0.0',
                         bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)
    conv1 = residual_unit(data=conv0, num_filter=64, stride=(2, 2), dim_match=False, name='conv1.0',
                          bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)
    conv2 = residual_unit(data=conv1, num_filter=128, stride=(2, 2), dim_match=False, name='conv2.0',
                          bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)

    conv1_img1, conv2_img1 = conv_unit(img1, 'img1', weights, bias)
    conv1_img2, conv2_img2 = conv_unit(img2, 'img2', weights, bias)

    corr = mx.sym.Correlation1D(data1=conv2_img1, data2=conv2_img2, pad_size=64, kernel_size=1,
                                max_displacement=64, stride1=1, stride2=1, single_side=-1)

    concat = mx.sym.Concat(corr, conv2)

    conv3a = residual_unit(data=concat, num_filter=256, stride=(2, 2), dim_match=False, name='conv3a',
                           bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)

    conv3b = residual_unit(data=conv3a, num_filter=256, stride=(1, 1), dim_match=False, name='conv3b',
                           bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)

    conv4a = residual_unit(data=conv3b, num_filter=512, stride=(2, 2), dim_match=False, name='conv4a',
                           bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)

    conv4b = residual_unit(data=conv4a, num_filter=512, stride=(1, 1), dim_match=False, name='conv4b',
                           bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)

    conv5a = residual_unit(data=conv4b, num_filter=512, stride=(2, 2), dim_match=False, name='conv5a',
                           bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)

    conv5b = residual_unit(data=conv5a, num_filter=512, stride=(1, 1), dim_match=False, name='conv5b',
                           bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)

    conv6a = residual_unit(data=conv5b, num_filter=1024, stride=(2, 2), dim_match=False, name='conv6a',
                           bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)

    conv6b = residual_unit(data=conv6a, num_filter=1024, stride=(1, 1), dim_match=False, name='conv6b',
                           bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)

    pr6 = mx.sym.Convolution(conv6b, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr6')

    upsample_pr6to5 = mx.sym.Deconvolution(pr6, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_pr6to5', no_bias=True)
    upconv5 = mx.sym.Deconvolution(conv6b, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=512, name='upconv5',
                                   no_bias=True)
    upconv5 = mx.sym.LeakyReLU(data=upconv5, act_type='leaky', slope=0.1)
    concat_tmp = mx.sym.Concat(conv5b, upconv5, upsample_pr6to5, dim=1)

    iconv5 = mx.sym.Convolution(concat_tmp, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name='iconv5')

    pr5 = mx.sym.Convolution(iconv5, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr5')

    upconv4 = mx.sym.Deconvolution(iconv5, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=256, name='upconv4',
                                   no_bias=True)
    upconv4 = mx.sym.LeakyReLU(data=upconv4, act_type='leaky', slope=0.1)

    upsample_pr5to4 = mx.sym.Deconvolution(pr5, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_pr5to4', no_bias=True)

    concat_tmp2 = mx.sym.Concat(conv4b, upconv4, upsample_pr5to4)
    iconv4 = mx.sym.Convolution(concat_tmp2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256, name='iconv4')
    pr4 = mx.sym.Convolution(iconv4, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr4')

    upconv3 = mx.sym.Deconvolution(iconv4, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=128, name='upconv3',
                                   no_bias=True)
    upconv3 = mx.sym.LeakyReLU(data=upconv3, act_type='leaky', slope=0.1)

    upsample_pr4to3 = mx.sym.Deconvolution(pr4, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_pr4to3', no_bias=True)
    concat_tmp3 = mx.sym.Concat(conv3b, upconv3, upsample_pr4to3)
    iconv3 = mx.sym.Convolution(concat_tmp3, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=128, name='iconv3')
    pr3 = mx.sym.Convolution(iconv3, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr3')

    upconv2 = mx.sym.Deconvolution(iconv3, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=64, name='upconv2',
                                   no_bias=True)
    upconv2 = mx.sym.LeakyReLU(data=upconv2, act_type='leaky', slope=0.1)

    upsample_pr3to2 = mx.sym.Deconvolution(pr3, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_pr3to2', no_bias=True)

    concat_tmp4 = mx.sym.Concat(conv2, upconv2, upsample_pr3to2)

    iconv2 = mx.sym.Convolution(concat_tmp4, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=64, name='iconv2')
    pr2 = mx.sym.Convolution(iconv2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr2')

    upconv1 = mx.sym.Deconvolution(iconv2, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=32, name='upconv1',
                                   no_bias=True)
    upconv1 = mx.sym.LeakyReLU(data=upconv1, act_type='leaky', slope=0.1)

    upsample_pr2to1 = mx.sym.Deconvolution(pr2, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_pr2to1', no_bias=True)

    concat_tmp5 = mx.sym.Concat(upconv1, upsample_pr2to1, conv1)
    iconv1 = mx.sym.Convolution(concat_tmp5, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=32, name='iconv1')
    pr1 = mx.sym.Convolution(iconv1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr1')
    pr = mx.sym.Deconvolution(pr1, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=1, name='init_label.drr',
                              no_bias=True)
    data = mx.sym.Concat(img1, img2)
    pr_final, map, replace_value, refine_value = detect_replace_refine(init_label=pr, input_feature=data,
                                                                     output_dim=output_dim, name='drr')
    # loss  = get_loss(pr_final, downsample0, 0.01, name='loss', get_data=False, is_sparse=is_sparse, type=net_type)
    loss0 = get_loss(pr, downsample0, 0.85, name='loss0', get_data=False, is_sparse=is_sparse, type=net_type)
    loss1 = get_loss(pr1, downsample1, 0.10, name='loss1', get_data=False, is_sparse=is_sparse, type=net_type)
    loss2 = get_loss(pr2, downsample2, 0.05, name='loss2', get_data=False, is_sparse=is_sparse, type=net_type)
    loss3 = get_loss(pr3, downsample3, loss3_scale, name='loss3', get_data=False, is_sparse=is_sparse, type=net_type)
    loss4 = get_loss(pr4, downsample4, loss4_scale, name='loss4', get_data=False, is_sparse=is_sparse, type=net_type)
    loss5 = get_loss(pr5, downsample5, loss5_scale, name='loss5', get_data=False, is_sparse=is_sparse, type=net_type)
    loss6 = get_loss(pr6, downsample6, loss6_scale, name='loss6', get_data=False, is_sparse=is_sparse, type=net_type)

    net = mx.sym.Group([loss0,loss1,loss2])

    return net



