from symbol_util import *

def detect(data, output_dim, name):

    data = get_conv_bn(name='detect.0'+name, data=data, num_filter=32, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9)
    data = mx.sym.Pooling(data=data,kernel=(2, 2),pool_type='max',stride=(2,2),name='detect.pool.0'+name)
    tmp2 = data = get_conv_bn(name='detect.1'+name, data=data, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9)
    data = mx.sym.Pooling(data=data, kernel=(2, 2), pool_type='max', stride=(2, 2), name='detect.pool.1'+name)
    data = get_conv_bn(name='detect.2'+name, data=data, num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9)
    data = get_conv_bn(name='detect.3'+name, data=data, num_filter=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9)
    data = get_conv_bn(name='detect.4'+name, data=data, num_filter=64, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + tmp2
    data = get_conv_bn(name='detect.5'+name, data=data, num_filter=32, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = get_conv_bn(name='detect.6'+name, data=data, num_filter=output_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=False, bn_momentum=0.9, is_conv=True)
    data = mx.sym.Activation(data=data, act_type='sigmoid', name='predict_error_map'+name)

    return data

def replace(data, output_dim, name):

    encoder0 = data = residual_unit(data=data, num_filter=64, stride=(2, 2), dim_match=False, name='replace.0'+name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    encoder1 = data = residual_unit(data=data, num_filter=128, stride=(2, 2), dim_match=False, name='replace.1'+name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    encoder2 = data = residual_unit(data=data, num_filter=256, stride=(2, 2), dim_match=False, name='replace.2'+name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    encoder3 = data = residual_unit(data=data, num_filter=512, stride=(2, 2), dim_match=False, name='replace.3'+name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    encoder4 = data = residual_unit(data=data, num_filter=512, stride=(2, 2), dim_match=False, name='replace.4'+name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    data = residual_unit(data=data, num_filter=1024, stride=(2, 2), dim_match=False, name='replace.5'+name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    data = get_conv_bn(name='replace.6'+name, data=data, num_filter=512, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder4
    data = get_conv_bn(name='replace.7'+name, data=data, num_filter=512, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder3
    data = get_conv_bn(name='replace.8'+name, data=data, num_filter=256, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder2
    data = get_conv_bn(name='replace.9'+name, data=data, num_filter=128, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder1
    data = get_conv_bn(name='replace.10'+name, data=data, num_filter=64, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder0
    data = get_conv_bn(name='replace.11'+name, data=data, num_filter=output_dim, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    return data

def refine(data, output_dim, name):

    data = residual_unit(data=data, num_filter=64, stride=(2, 2), dim_match=False, name='refine.0.0' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    encoder0 = data = residual_unit(data=data, num_filter=64, stride=(1, 1), dim_match=False, name='refine.0.1' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)

    data = residual_unit(data=data, num_filter=128, stride=(2, 2), dim_match=False, name='refine.1.0' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    encoder1 = data = residual_unit(data=data, num_filter=128, stride=(1, 1), dim_match=False, name='refine.1.1' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)

    data = residual_unit(data=data, num_filter=256, stride=(2, 2), dim_match=False, name='refine.2.0' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    encoder2 = data = residual_unit(data=data, num_filter=256, stride=(1, 1), dim_match=False, name='refine.2.1' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)

    data = residual_unit(data=data, num_filter=512, stride=(2, 2), dim_match=False, name='refine.3.0' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    data = residual_unit(data=data, num_filter=512, stride=(1, 1), dim_match=False, name='refine.3.1' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)

    data = get_conv_bn(name='refine.decoder0' + name, data=data, num_filter=256, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
             with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder2
    data = get_conv_bn(name='refine.decoder1' + name, data=data, num_filter=128, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder1
    data = get_conv_bn(name='refine.decoder2' + name, data=data, num_filter=64, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder0
    data = get_conv_bn(name='refine.decoder3' + name, data=data, num_filter=output_dim, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
             with_relu=False, bn_momentum=0.9, is_conv=False)
    return data


def detect_replace_refine(init_label, input_feature, output_dim, name):

    # detect error map
    data = mx.sym.Concat(init_label, input_feature)
    map = detect(data, output_dim, name)

    # replace
    data = mx.sym.Concat(data, map)
    replace_value = replace(data, output_dim, name)
    U = map * replace_value + (1-map) * init_label

    # refine
    data = mx.sym.Concat(data, U)
    refine_value = refine(data, output_dim, name)
    U = U + refine_value

    return U, map, replace_value, refine_value

def DRR_Dispnet(loss_scale, net_type='stereo', is_sparse = False):
    """
        create Dispnet or Flownet symbol. There is a slight difference between Dispnet and Flownet

        Parameters
        ----------
        loss_scale : dict of loss_scale,
            Dispnet and Flownet have six loss functions which have different sizes and loss scale.
            Example :
                {'loss1': 1.00, 'loss2': 0.00, 'loss3': 0.00, 'loss4':0.00, 'loss5':0.00,'loss6':0.00}
            'loss1' denotes the loss function which has the largest size
        net_type : str
            Should be 'stereo' or 'flow', default is 'stereo'
        is_sparse : bool
            indiate whether label contains NaN, default is False
            if the labels are sparse, it will call SparseRegressionLoss, Otherwise it use MAERegressionLoss

        Returns
        ----------
        net : symbol
            dispnet or flownet

        References
        ----------
            [1] A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation
            [2] FlowNet: Learning Optical Flow with Convolutional Networks
    """
    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    if net_type == 'stereo':
        output_dim = 1
    elif net_type == 'flow':
        output_dim = 2

    # six loss functions with different output sizes
    labels = {'loss{}'.format(i): mx.sym.Variable('loss{}_label'.format(i)) for i in range(0, 7)}
    prediction = {}
    loss = []

    # siamese network, two branches share the same weights
    weights = [mx.sym.Variable('share{}_weight'.format(i)) for i in range(1, 4)]
    bias = [mx.sym.Variable('share{}_bias'.format(i)) for i in range(1, 4)]
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
        conv_redir = mx.sym.Convolution(data=conv3_img1, pad=(0, 0), kernel=(1, 1), stride=(1, 1), num_filter=64,
                                        name='conv_redir')
        conv_redir = mx.sym.LeakyReLU(data=conv_redir, act_type='leaky', slope=0.1)
        concat = mx.sym.Concat(corr, conv_redir)

    if net_type == 'stereo':
        stride = (2, 2)
    elif net_type == 'flow':
        stride = (1, 1)

    # The structure below is similar to VGG
    conv3a = mx.sym.Convolution(concat, pad=(2, 2), kernel=(5, 5), stride=stride, num_filter=256, name='conv3a')
    conv3a = mx.sym.LeakyReLU(data=conv3a, act_type='leaky', slope=0.1)

    conv3b = mx.sym.Convolution(conv3a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256, name='conv3b')
    conv3b = mx.sym.LeakyReLU(data=conv3b, act_type='leaky', slope=0.1)

    conv4a = mx.sym.Convolution(conv3b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=512, name='conv4a')
    conv4a = mx.sym.LeakyReLU(data=conv4a, act_type='leaky', slope=0.1)

    conv4b = mx.sym.Convolution(conv4a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name='conv4b')
    conv4b = mx.sym.LeakyReLU(data=conv4b, act_type='leaky', slope=0.1)

    conv5a = mx.sym.Convolution(conv4b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=512, name='conv5a')
    conv5a = mx.sym.LeakyReLU(data=conv5a, act_type='leaky', slope=0.1)

    conv5b = mx.sym.Convolution(conv5a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name='conv5b')
    conv5b = mx.sym.LeakyReLU(data=conv5b, act_type='leaky', slope=0.1)

    conv6a = mx.sym.Convolution(conv5b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=1024, name='conv6a')
    conv6a = mx.sym.LeakyReLU(data=conv6a, act_type='leaky', slope=0.1)

    conv6b = mx.sym.Convolution(conv6a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=1024, name='conv6b')
    conv6b = mx.sym.LeakyReLU(data=conv6b, act_type='leaky', slope=0.1, )

    pr6 = mx.sym.Convolution(conv6b, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr6')
    prediction['loss6'] = pr6

    upsample_pr6to5 = mx.sym.Deconvolution(pr6, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_pr6to5', no_bias=True)
    upconv5 = mx.sym.Deconvolution(conv6b, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=512, name='upconv5',
                                   no_bias=True)
    upconv5 = mx.sym.LeakyReLU(data=upconv5, act_type='leaky', slope=0.1)
    concat_tmp = mx.sym.Concat(conv5b, upconv5, upsample_pr6to5, dim=1)

    iconv5 = mx.sym.Convolution(concat_tmp, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name='iconv5')

    pr5 = mx.sym.Convolution(iconv5, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr5')
    prediction['loss5'] = pr5

    upconv4 = mx.sym.Deconvolution(iconv5, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=256, name='upconv4',
                                   no_bias=True)
    upconv4 = mx.sym.LeakyReLU(data=upconv4, act_type='leaky', slope=0.1)

    upsample_pr5to4 = mx.sym.Deconvolution(pr5, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_pr5to4', no_bias=True)

    concat_tmp2 = mx.sym.Concat(conv4b, upconv4, upsample_pr5to4)
    iconv4 = mx.sym.Convolution(concat_tmp2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256,
                                name='iconv4')
    pr4 = mx.sym.Convolution(iconv4, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr4')
    prediction['loss4'] = pr4

    upconv3 = mx.sym.Deconvolution(iconv4, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=128, name='upconv3',
                                   no_bias=True)
    upconv3 = mx.sym.LeakyReLU(data=upconv3, act_type='leaky', slope=0.1)

    upsample_pr4to3 = mx.sym.Deconvolution(pr4, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_pr4to3', no_bias=True)
    concat_tmp3 = mx.sym.Concat(conv3b, upconv3, upsample_pr4to3)
    iconv3 = mx.sym.Convolution(concat_tmp3, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=128,
                                name='iconv3')
    pr3 = mx.sym.Convolution(iconv3, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr3')
    prediction['loss3'] = pr3

    upconv2 = mx.sym.Deconvolution(iconv3, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=64, name='upconv2',
                                   no_bias=True)
    upconv2 = mx.sym.LeakyReLU(data=upconv2, act_type='leaky', slope=0.1)

    upsample_pr3to2 = mx.sym.Deconvolution(pr3, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_pr3to2', no_bias=True)
    concat_tmp4 = mx.sym.Concat(conv2_img1, upconv2, upsample_pr3to2)
    iconv2 = mx.sym.Convolution(concat_tmp4, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=64, name='iconv2')
    pr2 = mx.sym.Convolution(iconv2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr2')
    prediction['loss2'] = pr2

    upconv1 = mx.sym.Deconvolution(iconv2, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=32, name='upconv1',
                                   no_bias=True)
    upconv1 = mx.sym.LeakyReLU(data=upconv1, act_type='leaky', slope=0.1)
    upsample_pr2to1 = mx.sym.Deconvolution(pr2, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_pr2to1', no_bias=True)
    concat_tmp5 = mx.sym.Concat(conv1_img1, upconv1, upsample_pr2to1)
    iconv1 = mx.sym.Convolution(concat_tmp5, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=32, name='iconv1')
    pr1 = mx.sym.Convolution(iconv1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr1')
    prediction['loss1'] = pr1

    # DRR
    init_label = mx.sym.UpSampling(data=prediction['loss1'], scale=2, num_filter=output_dim,
                             num_args=1, sample_type='bilinear', name='upsample_pr1')
    # init_label = mx.sym.BlockGrad(data=init_label, name='block_init_prediction')

    input_feature = mx.sym.Concat(img1, img2)
    DRR_prediction, map, replace_value, refine_value = detect_replace_refine(init_label, input_feature, output_dim, name='DRR')
    prediction['loss0'] = DRR_prediction

    # map = mx.sym.BlockGrad(data=map, name='map_block')
    # replace = mx.sym.BlockGrad(data=replace_value, name='replace_block')
    # refine = mx.sym.BlockGrad(data=refine_value, name='refine_block')
    # img = mx.sym.BlockGrad(data=img1, name='img1_block')

    # ignore the loss functions with loss scale of zero
    keys = loss_scale.keys()
    keys.sort()
    for key in keys:
        if loss_scale[key] > 0.0:
            loss.append(get_loss(-prediction[key], labels[key], loss_scale[key], name=key,
                                 get_input=False, is_sparse=is_sparse, type=net_type))

    # loss.extend([map, replace, refine, img])
    net = mx.sym.Group(loss)
    return net



