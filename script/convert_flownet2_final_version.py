
# coding: utf-8

# In[1]:

import mxnet as mx

from flow_stereo.symbol.symbol_util import *
from flow_stereo.data import dataset

def warp_flownet(img1, img2, flow, name, factor=4):
    
    flow = flow * 20.0
    flow = mx.sym.UpSampling(arg0=flow, scale=factor, num_filter=2,
                             num_args=1, sample_type='nearest', name='upsamplingop_flow{}'.format(name))
    img2_warped = warp(img=img2, flow=flow, name='nearest'.format(name))
    error = mx.sym.square(img1 - img2_warped)
    error = mx.sym.sum(error, axis=1, keepdims = True)
    error = mx.sym.sqrt(error)
    flow = flow * 0.05
    data = mx.sym.Concat(img1, img2, img2_warped, flow, error)
    
    return data

def warp_sd(img1, img2, flow_sd, flow_net3, name, factor=4):
    
    flow_sd = flow_sd * 0.05
    flow_sd = mx.sym.UpSampling(arg0=flow_sd, scale=factor, num_filter=2,
                             num_args=1, sample_type='nearest', name='upsamplingop_flow_sd{}'.format(name))
    flow_net3 = flow_net3 * 20
    flow_net3 = mx.sym.UpSampling(arg0=flow_net3, scale=factor, num_filter=2,
                             num_args=1, sample_type='nearest', name='upsamplingop_flow_net3'.format(name))
    blob164 = mx.sym.square(flow_sd)
    blob164 = mx.sym.sum(blob164, axis=1, keepdims = True)
    blob164 = mx.sym.sqrt(blob164)
    
    blob165 = mx.sym.square(flow_net3)
    blob165 = mx.sym.sum(blob165, axis=1, keepdims = True)
    blob165 = mx.sym.sqrt(blob165)
    
    img2_warped_sd = warp(img=img2, flow=flow_sd, name='flownet-{}-warp_sd'.format(name))
    
    error_sd = mx.sym.square(img1 - img2_warped_sd)
    error_sd = mx.sym.sum(error_sd, axis=1, keepdims = True)
    error_sd = mx.sym.sqrt(error_sd)
        
    img2_warped_net3 = warp(img=img2, flow=flow_net3, name='flownet-{}-warp_net3'.format(name))
    
    error_net3 = mx.sym.square(img1 - img2_warped_net3)
    error_net3 = mx.sym.sum(error_net3, axis=1, keepdims = True)
    error_net3 = mx.sym.sqrt(error_net3)
        
    data = mx.sym.concat(img1, flow_sd, flow_net3, blob164, blob165, error_sd, error_net3)
    return data
    
def flownet_s(data, labels, loss_scale, net_type='flow', is_sparse = False, name='flownet-s'):

    if net_type == 'stereo':
        output_dim = 1
    elif net_type == 'flow':
        output_dim = 2

    prediction = {}
    loss = []

    # The structure below is similar to VGG
    conv1 = mx.sym.Convolution(data, pad=(3, 3), kernel=(7, 7), stride=(2, 2), num_filter=64, name=name+'conv1')
    conv1 = mx.sym.LeakyReLU(data=conv1, act_type='leaky', slope=0.1)

    conv2 = mx.sym.Convolution(conv1, pad=(2, 2), kernel=(5, 5), stride=(2, 2), num_filter=128, name=name+'conv2')
    conv2 = mx.sym.LeakyReLU(data=conv2, act_type='leaky', slope=0.1)

    conv3a = mx.sym.Convolution(conv2, pad=(2, 2), kernel=(5, 5), stride=(2, 2), num_filter=256, name=name+'conv3')
    conv3a = mx.sym.LeakyReLU(data=conv3a, act_type='leaky', slope=0.1)

    conv3b = mx.sym.Convolution(conv3a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256, name=name+'conv3_1')
    conv3b = mx.sym.LeakyReLU(data=conv3b, act_type='leaky', slope=0.1)

    conv4a = mx.sym.Convolution(conv3b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=512, name=name+'conv4')
    conv4a = mx.sym.LeakyReLU(data=conv4a, act_type='leaky', slope=0.1)

    conv4b = mx.sym.Convolution(conv4a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name=name+'conv4_1')
    conv4b = mx.sym.LeakyReLU(data=conv4b, act_type='leaky', slope=0.1)

    conv5a = mx.sym.Convolution(conv4b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=512, name=name+'conv5')
    conv5a = mx.sym.LeakyReLU(data=conv5a, act_type='leaky', slope=0.1)

    conv5b = mx.sym.Convolution(conv5a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name=name+'conv5_1')
    conv5b = mx.sym.LeakyReLU(data=conv5b, act_type='leaky', slope=0.1)

    conv6a = mx.sym.Convolution(conv5b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=1024, name=name+'conv6')
    conv6a = mx.sym.LeakyReLU(data=conv6a, act_type='leaky', slope=0.1)

    conv6b = mx.sym.Convolution(conv6a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=1024, name=name+'conv6_1')
    conv6b = mx.sym.LeakyReLU(data=conv6b, act_type='leaky', slope=0.1, )

    pr6 = mx.sym.Convolution(conv6b,pad= (1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name=name+'predict_conv6')
    prediction['loss6'] = pr6

    upsample_pr6to5 = mx.sym.Deconvolution(pr6, pad=(1,1), kernel=(4,4), stride=(2,2), num_filter=output_dim,
                                           name=name+name+'upsample_flow6to5',no_bias=True)
    upconv5 = mx.sym.Deconvolution(conv6b,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=512,name=name+'deconv5',no_bias=True)
    upconv5 = mx.sym.LeakyReLU(data = upconv5,act_type = 'leaky',slope  = 0.1)
    concat_tmp = mx.sym.Concat(conv5b,upconv5,upsample_pr6to5,dim=1)
    
    if net_type == 'stereo':
        iconv5 = mx.sym.Convolution(concat_tmp,pad = (1,1),kernel=(3,3),stride=(1,1),num_filter = 512,name=name+'iconv5')
    else:
        iconv5 = concat_tmp
    pr5  = mx.sym.Convolution(iconv5, pad = (1,1),kernel=(3,3),stride=(1,1),num_filter = output_dim,name=name+'predict_conv5')
    prediction['loss5'] = pr5

    upconv4 = mx.sym.Deconvolution(iconv5,pad = (1,1),kernel= (4,4),stride = (2,2),num_filter=256,name=name+'deconv4',no_bias=True)
    upconv4 = mx.sym.LeakyReLU(data = upconv4,act_type = 'leaky',slope  = 0.1 )

    upsample_pr5to4 = mx.sym.Deconvolution(pr5,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name=name + name+'upsample_flow5to4',no_bias=True)

    concat_tmp2 = mx.sym.Concat(conv4b,upconv4,upsample_pr5to4)
    if net_type == 'stereo':
        iconv4  = mx.sym.Convolution(concat_tmp2,pad = (1,1),kernel = (3,3),stride=(1,1),num_filter=256,name=name+'iconv4')
    else:
        iconv4 = concat_tmp2
    pr4 = mx.sym.Convolution(iconv4,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name=name+'predict_conv4')
    prediction['loss4'] = pr4

    upconv3 = mx.sym.Deconvolution(iconv4,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=128,name=name+'deconv3',no_bias=True)
    upconv3 = mx.sym.LeakyReLU(data = upconv3,act_type = 'leaky',slope  = 0.1 )

    upsample_pr4to3 = mx.sym.Deconvolution(pr4,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name=name+name+'upsample_flow4to3',no_bias=True)
    concat_tmp3 = mx.sym.Concat(conv3b,upconv3,upsample_pr4to3)
    if net_type == 'stereo':
        iconv3 = mx.sym.Convolution(concat_tmp3,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter = 128,name=name+'iconv3')
    else:
        iconv3 = concat_tmp3
    pr3 = mx.sym.Convolution(iconv3,pad = (1,1), kernel = (3,3), stride = (1,1),num_filter = output_dim,name=name+'predict_conv3')
    prediction['loss3'] = pr3

    upconv2 = mx.sym.Deconvolution(iconv3,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=64,name=name+'deconv2',no_bias=True)
    upconv2 = mx.sym.LeakyReLU(data = upconv2,act_type = 'leaky',slope  = 0.1  )

    upsample_pr3to2 = mx.sym.Deconvolution(pr3,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name=name + name+'upsample_flow3to2',no_bias=True)
    concat_tmp4 = mx.sym.Concat(conv2, upconv2, upsample_pr3to2)
    if net_type == 'stereo':
        iconv2 = mx.sym.Convolution(concat_tmp4,pad = (1,1),kernel = (3,3),stride= (1,1),num_filter = 64,name=name+'iconv2')
    else:
        iconv2 = concat_tmp4
    pr2 = mx.sym.Convolution(iconv2,pad = (1,1),kernel=(3,3),stride = (1,1),num_filter = output_dim,name=name+'predict_conv2')
    prediction['loss2'] = pr2

    # ignore the loss functions with loss scale of zero
    keys = loss_scale.keys()
    keys.sort()
    for key in keys:
        loss.append(get_loss(prediction[key]*20, labels[key], loss_scale[key], name=key+name,
                             get_input=False, is_sparse = is_sparse, type=net_type))
    return prediction, loss

def flownet_c(img1, img2, labels, loss_scale, net_type='flow', is_sparse=False):

    if net_type == 'stereo':
        output_dim = 1
    elif net_type == 'flow':
        output_dim = 2

    prediction = {}
    loss = []

    # siamese network, two branches share the same weights
    weights = [mx.sym.Variable('conv{}_weight'.format(i)) for i in range(1, 4)]
    bias = [mx.sym.Variable('conv{}_bias'.format(i)) for i in range(1, 4)]
    conv1_img1, conv2_img1 = conv_share(img1, 'img1', weights, bias)
    conv1_img2, conv2_img2 = conv_share(img2, 'img2', weights, bias)
    
    # difference between DispNet and FlowNet
    if net_type == 'stereo':
        corr = mx.sym.Correlation1D(data1=conv2_img1, data2=conv2_img2, pad_size=40, kernel_size=1,
                                    max_displacement=40, stride1=1, stride2=1)
        conv_redir = mx.sym.Convolution(data=conv2_img1, pad=(0, 0), kernel=(1, 1), stride=(1, 1), num_filter=64,
                                        name='conv_redir')
        conv_redir = mx.sym.LeakyReLU(data=conv_redir, act_type='leaky', slope=0.1)
        concat = mx.sym.Concat(corr, conv_redir)
    elif net_type == 'flow':
        conv3_img1 = mx.sym.Convolution(data=conv2_img1, pad=(2, 2), kernel=(5, 5), stride=(2, 2), num_filter=256,
                                        weight=weights[2], bias=bias[2], name='conv3_img1')
        conv3_img1 = mx.sym.LeakyReLU(data=conv3_img1, act_type='leaky', slope=0.1)

        conv3_img2 = mx.sym.Convolution(data=conv2_img2, pad=(2, 2), kernel=(5, 5), stride=(2, 2), num_filter=256,
                                        weight=weights[2], bias=bias[2], name='conv3_img2')
        conv3_img2 = mx.sym.LeakyReLU(data=conv3_img2, act_type='leaky', slope=0.1)

        corr = mx.sym.Correlation(data1=conv3_img1, data2=conv3_img2, pad_size=20, kernel_size=1,
                                  max_displacement=20, stride1=1, stride2=2)
        corr = mx.sym.LeakyReLU(data=corr, act_type='leaky', slope=0.1)
        conv_redir = mx.sym.Convolution(data=conv3_img1, pad=(0, 0), kernel=(1, 1), stride=(1, 1), num_filter=32,
                                        name='conv_redir')
        conv_redir = mx.sym.LeakyReLU(data=conv_redir, act_type='leaky', slope=0.1)
        concat = mx.sym.Concat(conv_redir, corr)

    if net_type == 'stereo':
        stride = (2, 2)
    elif net_type == 'flow':
        stride = (1, 1)
    
    # The structure below is similar to VGG
    if net_type == 'stereo':
        conv3a = mx.sym.Convolution(concat, pad=(2, 2), kernel=(5, 5), stride=stride, num_filter=256, name='conv3')
        conv3a = mx.sym.LeakyReLU(data=conv3a, act_type='leaky', slope=0.1)
    else:
        conv3a = concat
    
#     data_tmp = mx.sym.BlockGrad(data=concat, name='corr_conv')
    conv3b = mx.sym.Convolution(conv3a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256, name='conv3_1')
    conv3b = mx.sym.LeakyReLU(data=conv3b, act_type='leaky', slope=0.1)

    conv4a = mx.sym.Convolution(conv3b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=512, name='conv4')
    conv4a = mx.sym.LeakyReLU(data=conv4a, act_type='leaky', slope=0.1)

    conv4b = mx.sym.Convolution(conv4a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name='conv4_1')
    conv4b = mx.sym.LeakyReLU(data=conv4b, act_type='leaky', slope=0.1)

    conv5a = mx.sym.Convolution(conv4b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=512, name='conv5')
    conv5a = mx.sym.LeakyReLU(data=conv5a, act_type='leaky', slope=0.1)

    conv5b = mx.sym.Convolution(conv5a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name='conv5_1')
    conv5b = mx.sym.LeakyReLU(data=conv5b, act_type='leaky', slope=0.1)

    conv6a = mx.sym.Convolution(conv5b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=1024, name='conv6')
    conv6a = mx.sym.LeakyReLU(data=conv6a, act_type='leaky', slope=0.1)

    conv6b = mx.sym.Convolution(conv6a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=1024, name='conv6_1')
    conv6b = mx.sym.LeakyReLU(data=conv6b, act_type='leaky', slope=0.1, )

    pr6 = mx.sym.Convolution(conv6b, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='Convolution1')
    prediction['loss6'] = pr6

    upsample_pr6to5 = mx.sym.Deconvolution(pr6, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_flow6to5', no_bias=True)
    upconv5 = mx.sym.Deconvolution(conv6b, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=512, name='deconv5',
                                   no_bias=True)
    upconv5 = mx.sym.LeakyReLU(data=upconv5, act_type='leaky', slope=0.1)
    concat_tmp = mx.sym.Concat(conv5b, upconv5, upsample_pr6to5, dim=1)
    
    if net_type == 'stereo':
        iconv5 = mx.sym.Convolution(concat_tmp, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name='iconv5')
    else:
        iconv5 = concat_tmp

    pr5 = mx.sym.Convolution(iconv5, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='Convolution2')
    prediction['loss5'] = pr5

    upconv4 = mx.sym.Deconvolution(iconv5, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=256, name='deconv4',
                                   no_bias=True)
    upconv4 = mx.sym.LeakyReLU(data=upconv4, act_type='leaky', slope=0.1)

    upsample_pr5to4 = mx.sym.Deconvolution(pr5, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_flow5to4', no_bias=True)

    concat_tmp2 = mx.sym.Concat(conv4b, upconv4, upsample_pr5to4)
    if net_type == 'stereo':
        iconv4 = mx.sym.Convolution(concat_tmp2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256, name='iconv4')
    else:
        iconv4 = concat_tmp2
    pr4 = mx.sym.Convolution(iconv4, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='Convolution3')
    prediction['loss4'] = pr4

    upconv3 = mx.sym.Deconvolution(iconv4, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=128, name='deconv3',
                                   no_bias=True)
    upconv3 = mx.sym.LeakyReLU(data=upconv3, act_type='leaky', slope=0.1)

    upsample_pr4to3 = mx.sym.Deconvolution(pr4, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_flow4to3', no_bias=True)
    concat_tmp3 = mx.sym.Concat(conv3b, upconv3, upsample_pr4to3)
    if net_type == 'stereo':
        iconv3 = mx.sym.Convolution(concat_tmp3, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=128, name='iconv3')
    else:
        iconv3 = concat_tmp3
    pr3 = mx.sym.Convolution(iconv3, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='Convolution4')
    prediction['loss3'] = pr3

    upconv2 = mx.sym.Deconvolution(iconv3, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=64, name='deconv2',
                                   no_bias=True)
    upconv2 = mx.sym.LeakyReLU(data=upconv2, act_type='leaky', slope=0.1)

    upsample_pr3to2 = mx.sym.Deconvolution(pr3, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_flow3to2', no_bias=True)
    concat_tmp4 = mx.sym.Concat(conv2_img1, upconv2, upsample_pr3to2)
    if net_type == 'stereo':
        iconv2 = mx.sym.Convolution(concat_tmp4, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=64, name='iconv2')
    else:
        iconv2 = concat_tmp4
    pr2 = mx.sym.Convolution(iconv2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='Convolution5')
    prediction['loss2'] = pr2

    # ignore the loss functions with loss scale of zero
    keys = loss_scale.keys()
    keys.sort()
    for key in keys:
        loss.append(get_loss(prediction[key]* 20, labels[key], loss_scale[key], name=key,
                             get_input=False, is_sparse=is_sparse, type=net_type))
#     loss.append(data_tmp)
    return prediction, loss

def netsd(img1, img2, labels, loss_scale, net_type='flow', is_sparse=False):
    
    prediction = {}
    loss = []
    
    data = mx.sym.concat(img1, img2)
    conv0 = mx.sym.Convolution(data, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=64, name='netsd_conv0')
    conv0 = mx.sym.LeakyReLU(data=conv0, act_type='leaky', slope=0.1)
    
    conv1 = mx.sym.Convolution(conv0, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=64, name='netsd_conv1')
    conv1 = mx.sym.LeakyReLU(data=conv1, act_type='leaky', slope=0.1)
    
    conv1_1 = mx.sym.Convolution(conv1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=128, name='netsd_conv1_1')
    conv1_1 = mx.sym.LeakyReLU(data=conv1_1, act_type='leaky', slope=0.1)
    
    conv2 = mx.sym.Convolution(conv1_1, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=128, name='netsd_conv2')
    conv2 = mx.sym.LeakyReLU(data=conv2, act_type='leaky', slope=0.1)
    
    conv2_1 = mx.sym.Convolution(conv2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=128, name='netsd_conv2_1')
    conv2_1 = mx.sym.LeakyReLU(data=conv2_1, act_type='leaky', slope=0.1)
    
    conv3 = mx.sym.Convolution(conv2_1, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=256, name='netsd_conv3')
    conv3 = mx.sym.LeakyReLU(data=conv3, act_type='leaky', slope=0.1)
    
    conv3_1 = mx.sym.Convolution(conv3, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256, name='netsd_conv3_1')
    conv3_1 = mx.sym.LeakyReLU(data=conv3_1, act_type='leaky', slope=0.1)
    
    conv4 = mx.sym.Convolution(conv3_1, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=512, name='netsd_conv4')
    conv4 = mx.sym.LeakyReLU(data=conv4, act_type='leaky', slope=0.1)
    
    conv4_1 = mx.sym.Convolution(conv4, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name='netsd_conv4_1')
    conv4_1 = mx.sym.LeakyReLU(data=conv4_1, act_type='leaky', slope=0.1)
    
    conv5 = mx.sym.Convolution(conv4_1, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=512, name='netsd_conv5')
    conv5 = mx.sym.LeakyReLU(data=conv5, act_type='leaky', slope=0.1)
    
    conv5_1 = mx.sym.Convolution(conv5, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name='netsd_conv5_1')
    conv5_1 = mx.sym.LeakyReLU(data=conv5_1, act_type='leaky', slope=0.1)
    
    conv6 = mx.sym.Convolution(conv5_1, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=1024, name='netsd_conv6')
    conv6 = mx.sym.LeakyReLU(data=conv6, act_type='leaky', slope=0.1)
    
    conv6_1 = mx.sym.Convolution(conv6, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=1024, name='netsd_conv6_1')
    conv6_1 = mx.sym.LeakyReLU(data=conv6_1, act_type='leaky', slope=0.1)
    
    prediction['loss6'] = netsd_Convolution1 = mx.sym.Convolution(conv6_1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=2, name='netsd_Convolution1')
    
    netsd_deconv5 = mx.sym.Deconvolution(conv6_1, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=512, name='netsd_deconv5', no_bias=True)
    netsd_deconv5 = mx.sym.LeakyReLU(data=netsd_deconv5, act_type='leaky', slope=0.1)
    
    netsd_upsample_flow6to5 = mx.sym.Deconvolution(netsd_Convolution1, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=2, 
                                                   name='netsd_upsample_flow6to5',no_bias=True)
    concat = mx.sym.Concat(conv5_1, netsd_deconv5, netsd_upsample_flow6to5)
    
    netsd_interconv5 = mx.sym.Convolution(concat, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name='netsd_interconv5')
    prediction['loss5'] = netsd_Convolution2 = mx.sym.Convolution(netsd_interconv5, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=2, name='netsd_Convolution2')
    
    netsd_deconv4 = mx.sym.Deconvolution(concat, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=256, name='netsd_deconv4',no_bias=True)
    netsd_deconv4 = mx.sym.LeakyReLU(data=netsd_deconv4, act_type='leaky', slope=0.1)
    
    netsd_upsample_flow5to4 = mx.sym.Deconvolution(netsd_Convolution2, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=2, 
                                                   name='netsd_upsample_flow5to4',no_bias=True)
    concat = mx.sym.Concat(conv4_1, netsd_deconv4, netsd_upsample_flow5to4)
    
    netsd_interconv4 = mx.sym.Convolution(concat, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256, name='netsd_interconv4')
    prediction['loss4'] = netsd_Convolution3 = mx.sym.Convolution(netsd_interconv4, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=2, name='netsd_Convolution3')
    
    netsd_deconv3 = mx.sym.Deconvolution(concat, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=128, name='netsd_deconv3',no_bias=True)
    netsd_deconv3 = mx.sym.LeakyReLU(data=netsd_deconv3, act_type='leaky', slope=0.1)
    
    netsd_upsample_flow4to3 = mx.sym.Deconvolution(netsd_Convolution3, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=2, 
                                                   name='netsd_upsample_flow4to3',no_bias=True)
    concat = mx.sym.Concat(conv3_1, netsd_deconv3, netsd_upsample_flow4to3)
    
    netsd_interconv3 = mx.sym.Convolution(concat, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=128, name='netsd_interconv3')
    prediction['loss3'] = netsd_Convolution4 = mx.sym.Convolution(netsd_interconv3, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=2, name='netsd_Convolution4')
    
    netsd_deconv2 = mx.sym.Deconvolution(concat, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=64, name='netsd_deconv2',no_bias=True)
    netsd_deconv2 = mx.sym.LeakyReLU(data=netsd_deconv2, act_type='leaky', slope=0.1)
    
    netsd_upsample_flow3to2 = mx.sym.Deconvolution(netsd_Convolution4, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=2, 
                                                   name='netsd_upsample_flow3to2',no_bias=True)
    concat = mx.sym.Concat(conv2_1, netsd_deconv2, netsd_upsample_flow3to2)
    
    netsd_interconv2 = mx.sym.Convolution(concat, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=64, name='netsd_interconv2')
    prediction['loss2'] = netsd_Convolution5 = mx.sym.Convolution(netsd_interconv2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=2, name='netsd_Convolution5')
    
    # ignore the loss functions with loss scale of zero
    keys = loss_scale.keys()
    keys.sort()
    for key in keys:
        loss.append(get_loss(prediction[key]* 20, labels[key], loss_scale[key], name=key,
                             get_input=False, is_sparse=is_sparse, type=net_type))
    return prediction, loss

def fusenet(data, labels, loss_scale, net_type='flow', is_sparse=False):
    
    prediction = {}
    loss = []
    
    fuse_conv0 = mx.sym.Convolution(data, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=64, name='fuse_conv0')
    fuse_conv0 = mx.sym.LeakyReLU(data=fuse_conv0, act_type='leaky', slope=0.1)
    
    fuse_conv1 = mx.sym.Convolution(data=fuse_conv0, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=64, name='fuse_conv1')
    fuse_conv1 = mx.sym.LeakyReLU(data=fuse_conv1, act_type='leaky', slope=0.1)
    
    fuse_conv1_1 = mx.sym.Convolution(fuse_conv1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=128, name='fuse_conv1_1')
    fuse_conv1_1 = mx.sym.LeakyReLU(data=fuse_conv1_1, act_type='leaky', slope=0.1)
    
    fuse_conv2 = mx.sym.Convolution(data=fuse_conv1_1, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=128, name='fuse_conv2')
    fuse_conv2 = mx.sym.LeakyReLU(data=fuse_conv2, act_type='leaky', slope=0.1)
    
    fuse_conv2_1 = mx.sym.Convolution(fuse_conv2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=128, name='fuse_conv2_1')
    fuse_conv2_1 = mx.sym.LeakyReLU(data=fuse_conv2_1, act_type='leaky', slope=0.1)
    
    fuse_Convolution5 = mx.sym.Convolution(fuse_conv2_1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=2, name='fuse__Convolution5')
    prediction['loss2'] = fuse_Convolution5
    fuse_deconv1 = mx.sym.Deconvolution(fuse_conv2_1, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=32, name='fuse_deconv1', no_bias=True)
    fuse_deconv1 = mx.sym.LeakyReLU(data=fuse_deconv1, act_type='leaky', slope=0.1)

    fuse_upsample_flow2to1 = mx.sym.Deconvolution(fuse_Convolution5, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=2, name='fuse_upsample_flow2to1', no_bias=True)
    concat = mx.sym.concat(fuse_conv1_1, fuse_deconv1, fuse_upsample_flow2to1)
    fuse_interconv1 =  mx.sym.Convolution(concat, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=32, name='fuse_interconv1')
    
    prediction['loss1'] = fuse_Convolution6 = mx.sym.Convolution(fuse_interconv1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=2, name='fuse__Convolution6')
    
    fuse_deconv0 = mx.sym.Deconvolution(concat, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=16, name='fuse_deconv0',no_bias=True)
    fuse_deconv0 = mx.sym.LeakyReLU(data=fuse_deconv0, act_type='leaky', slope=0.1)
    
    fuse_upsample_flow1to0 =  mx.sym.Deconvolution(fuse_Convolution6, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=2, name='fuse_upsample_flow1to0',no_bias=True)
    concat = mx.sym.concat(fuse_conv0, fuse_deconv0, fuse_upsample_flow1to0)
    fuse_interconv0 =  mx.sym.Convolution(concat, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=16, name='fuse_interconv0')
    prediction['loss0'] = fuse_Convolution7 = mx.sym.Convolution(fuse_interconv0, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=2, name='fuse__Convolution7')
    
    keys = loss_scale.keys()
    keys.sort()
    for key in keys:
        print key
        loss.append(get_loss(prediction[key], labels[key], loss_scale[key], name=key,
                             get_input=False, is_sparse=is_sparse, type=net_type))
    return prediction, loss
    
def flownet2(loss_scale, net_type='flow', is_sparse=False):

    # input
    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    # six loss functions with different output sizes
    labels = {'loss{}'.format(i): mx.sym.Variable('loss{}_label'.format(i)) for i in range(0, 7)}
    
    # flownet-c
    flownetc_prediction, flownetc_loss = flownet_c(img1, img2, labels, loss_scale['flownetc'],
                                                   net_type=net_type, is_sparse=is_sparse)
    flownetc_params = flownetc_loss[0].list_arguments()

    # flownet-s1
    data = warp_flownet(img1, img2, flow=flownetc_prediction['loss2'], name='s1')
    data_tmp = mx.sym.BlockGrad(data= data, name='block_s1_data')
    flownets1_prediction, flownets1_loss = flownet_s(data, labels, loss_scale['flownets1'],
                                                     net_type=net_type, is_sparse = is_sparse,
                                                     name='net2_')

    # flownet-s2
    data = warp_flownet(img1, img2, flow=flownets1_prediction['loss2'], name='s2')
    flownets2_prediction, flownets2_loss = flownet_s(data, labels, loss_scale['flownets2'],
                                                     net_type=net_type, is_sparse=is_sparse,
                                                     name='net3_')
    
    # SD
    sd_prediction, sd_loss = netsd(img1, img2, labels, loss_scale['sd'], net_type=net_type, is_sparse=is_sparse)
    
    # fuse net
    data = warp_sd(img1, img2, flow_sd=sd_prediction['loss2'], flow_net3=flownets2_prediction['loss2'], name='sd', factor=4)
    fuse_prediction, fuse_loss = fusenet(data, labels, loss_scale['fuse'], net_type=net_type, is_sparse=is_sparse)
    
    loss = fuse_loss
    loss.extend(sd_loss)
    # loss = sd_loss
    loss.extend(flownets2_loss)
    loss.extend(flownets1_loss)
    loss.extend(flownetc_loss)
    net = mx.sym.Group(loss)

    return net, flownetc_params


# ## Caffe

# In[2]:

import sys
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
prototxt = '/home/xudong/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt'
model = '/home/xudong/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5'

import caffe
import cv2

# caffe.set_device(1)
caffe.set_mode_gpu()
net = caffe.Net(prototxt, model, caffe.TEST)

name_caffe = net.params.keys()


# ## MXNet

# In[ ]:

loss_scale = {}
loss_scale['flownetc'] = {'loss2': 1.00,
                          'loss3': 1.00,
                          'loss4': 1.00,
                          'loss5': 1.00,
                          'loss6': 1.00}


loss_scale['flownets1'] = {'loss2': 1.00,
                          'loss3': 1.00,
                          'loss4': 1.00,
                          'loss5': 1.00,
                          'loss6': 1.00}

loss_scale['flownets2'] = {'loss2': 1.00,
                          'loss3': 1.00,
                          'loss4': 1.00,
                          'loss5': 1.00,
                          'loss6': 1.00}

loss_scale['sd'] =        {'loss2': 1.00,
                          'loss3': 1.00,
                          'loss4': 1.00,
                          'loss5': 1.00,
                          'loss6': 1.00}

loss_scale['fuse'] =      {'loss0': 1.00,
                           'loss1': 1.00,
                           'loss2': 1.00}
    
symbol, _  = flownet2(loss_scale, net_type='flow', is_sparse=False)

name_mxnet = symbol.list_arguments()

name_caffe_1 = []
for key in name_caffe:
    if 'img' not in key:
        name_caffe_1.append(key)

name_mxnet_1 = []
for key in name_mxnet:
    if 'img' not in key and 'bias' not in key and 'label' not in key:
        name_mxnet_1.append(key[:-7])

name_caffe_1.sort()
name_mxnet_1.sort()
# for i in range(len(name_mxnet_1)):
#     assert (name_caffe_1[i] == name_mxnet_1[i]),

shapes = (1, 3, 384, 512)
exe = symbol.simple_bind(ctx=mx.gpu(0), img1=shapes, img2=shapes)


# In[4]:

data = dataset.FlyingChairsDataset()
img1, img2, gt, _ = data.get_data(data.dirs[0])
input_dict = {}
input_dict['img0'] = img1[np.newaxis, :384, :768, :].transpose(0, 3, 1, 2)
input_dict['img1'] = img2[np.newaxis, :384, :768, :].transpose(0, 3, 1, 2)
net.forward(**input_dict)

for key in name_caffe_1:
    if key == 'scale_conv1':
        continue
    try:
        exe.arg_dict[key+'_weight'][:] = net.params[key][0].data
        if  key + '_bias' in exe.arg_dict:
            exe.arg_dict[key+'_bias'][:] = net.params[key][1].data
    except:
        print key, net.params[key][0].data.shape, exe.arg_dict[key+'_weight'].shape

# init upsampling
init = mx.initializer.Bilinear()
for key in exe.arg_dict:
    if 'upsamplingop' in key:
        init._init_weight(None, exe.arg_dict[key])

exe.arg_dict['img1'][:] = net.blobs['img0_nomean'].data
exe.arg_dict['img2'][:] = net.blobs['img1_nomean'].data
for key in exe.arg_dict:
    if exe.arg_dict[key].asnumpy().sum() == 0:
        print key

exe.forward()

mx.model.save_checkpoint(prefix='/rawdata/checkpoint_flowstereo/model_zoo/caffe_flownet2', 
                         epoch = 0, 
                         symbol=symbol, 
                         arg_params=exe.arg_dict, 
                         aux_params=exe.aux_dict)


# In[13]:

from flow_stereo.others import visualize

color = visualize.flow2color(exe.outputs[0].asnumpy()[0].transpose(1,2,0))
plt.imshow(color)

