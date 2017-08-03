from .symbol_util import *

def dispnet(loss_scale, net_type='stereo', is_sparse = False):
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
    labels = {'loss{}'.format(i) :  mx.sym.Variable('loss{}_label'.format(i)) for i in range(0, 7)}
    prediction = {}
    loss = []

    # siamese network, two branches share the same weights
    weights = [mx.sym.Variable('share{}_weight'.format(i)) for i in range(1,4)]
    bias    = [mx.sym.Variable('share{}_bias'.format(i)) for i in range(1,4)]
    conv1_img1, conv2_img1 = conv_share(img1, 'img1', weights, bias)
    conv1_img2, conv2_img2 = conv_share(img2, 'img2', weights, bias)

    # difference between DispNet and FlowNet
    if net_type =='stereo':
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

    if net_type =='stereo':
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

    pr6 = mx.sym.Convolution(conv6b,pad= (1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name='pr6')
    prediction['loss6'] = pr6

    upsample_pr6to5 = mx.sym.Deconvolution(pr6, pad=(1,1), kernel=(4,4), stride=(2,2), num_filter=output_dim,
                                           name='upsample_pr6to5',no_bias=True)
    upconv5 = mx.sym.Deconvolution(conv6b,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=512,name='upconv5',no_bias=True)
    upconv5 = mx.sym.LeakyReLU(data = upconv5,act_type = 'leaky',slope  = 0.1)
    concat_tmp = mx.sym.Concat(conv5b,upconv5,upsample_pr6to5,dim=1)

    iconv5 = mx.sym.Convolution(concat_tmp,pad = (1,1),kernel=(3,3),stride=(1,1),num_filter = 512,name='iconv5')

    pr5  = mx.sym.Convolution(iconv5, pad = (1,1),kernel=(3,3),stride=(1,1),num_filter = output_dim,name='pr5')
    prediction['loss5'] = pr5

    upconv4 = mx.sym.Deconvolution(iconv5,pad = (1,1),kernel= (4,4),stride = (2,2),num_filter=256,name='upconv4',no_bias=True)
    upconv4 = mx.sym.LeakyReLU(data = upconv4,act_type = 'leaky',slope  = 0.1 )

    upsample_pr5to4 = mx.sym.Deconvolution(pr5,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name='upsample_pr5to4',no_bias=True)

    concat_tmp2 = mx.sym.Concat(conv4b,upconv4,upsample_pr5to4)
    iconv4  = mx.sym.Convolution(concat_tmp2,pad = (1,1),kernel = (3,3),stride=(1,1),num_filter=256,name='iconv4')
    pr4 = mx.sym.Convolution(iconv4,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name='pr4')
    prediction['loss4'] = pr4

    upconv3 = mx.sym.Deconvolution(iconv4,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=128,name='upconv3',no_bias=True)
    upconv3 = mx.sym.LeakyReLU(data = upconv3,act_type = 'leaky',slope  = 0.1 )

    upsample_pr4to3 = mx.sym.Deconvolution(pr4,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name='upsample_pr4to3',no_bias=True)
    concat_tmp3 = mx.sym.Concat(conv3b,upconv3,upsample_pr4to3)
    iconv3 = mx.sym.Convolution(concat_tmp3,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter = 128,name='iconv3')
    pr3 = mx.sym.Convolution(iconv3,pad = (1,1), kernel = (3,3), stride = (1,1),num_filter = output_dim,name='pr3')
    prediction['loss3'] = pr3

    upconv2 = mx.sym.Deconvolution(iconv3,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=64,name='upconv2',no_bias=True)
    upconv2 = mx.sym.LeakyReLU(data = upconv2,act_type = 'leaky',slope  = 0.1  )

    upsample_pr3to2 = mx.sym.Deconvolution(pr3,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name='upsample_pr3to2',no_bias=True)
    concat_tmp4 = mx.sym.Concat(conv2_img1,upconv2,upsample_pr3to2)
    iconv2 = mx.sym.Convolution(concat_tmp4,pad = (1,1),kernel = (3,3),stride= (1,1),num_filter = 64,name='iconv2')
    pr2 = mx.sym.Convolution(iconv2,pad = (1,1),kernel=(3,3),stride = (1,1),num_filter = output_dim,name='pr2')
    prediction['loss2'] = pr2

    upconv1 = mx.sym.Deconvolution(iconv2,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter = 32,name='upconv1',no_bias=True)
    upconv1 = mx.sym.LeakyReLU(data = upconv1,act_type = 'leaky',slope  = 0.1 )
    upsample_pr2to1 = mx.sym.Deconvolution(pr2,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name='upsample_pr2to1',no_bias=True)
    concat_tmp5 = mx.sym.Concat(conv1_img1,upconv1,upsample_pr2to1)
    iconv1 = mx.sym.Convolution(concat_tmp5,pad=(1,1),kernel = (3,3),stride=(1,1),num_filter=32,name='iconv1')
    pr1 = mx.sym.Convolution(iconv1,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name='pr1')
    prediction['loss1'] = pr1

    # confidence
    data = warp_dispnet(img1=img1, img2=img2, disp=-pr1, name='warp', factor=2)
    confidence = detect(data, output_dim=1, name='detect')

    pr0 = mx.sym.UpSampling(data=-pr1, scale=2, num_filter=1,
                            num_args=1, sample_type='bilinear',
                            name='upsamplingop_pr')
    label_confidence = mx.sym.where(condition=mx.sym.abs(pr0-labels['loss0'])<=3,
                                    x=mx.sym.ones_like(data=pr0),
                                    y=mx.sym.zeros_like(data=pr0))
    scale = mx.sym.where(condition=label_confidence,
                         x=mx.sym.ones_like(data=pr0),
                         y=mx.sym.ones_like(data=pr0)*5)
    mae = mx.sym.abs(label_confidence-confidence) * scale
    cross_entropy = mx.sym.MakeLoss(data=mae)
    # ignore the loss functions with loss scale of zero
    keys = loss_scale.keys()
    keys.sort()
    for key in keys:
        if loss_scale[key] > 0.0:
            loss.append(get_loss(-prediction[key], labels[key], loss_scale[key], name=key,
                                 get_input=False, is_sparse = is_sparse, type=net_type))
    loss.append(cross_entropy)
    label_confidence = mx.sym.BlockGrad(data=label_confidence, name='block_label_tmp')
    confidence = mx.sym.BlockGrad(data=confidence, name='block_confidence')
    loss.append(confidence)
    loss.append(label_confidence)
    net = mx.sym.Group(loss)
    return net

# var_registrar = {}
# def get_variable(name, shape=None, init=None):
#     global var_registrar
#     if name not in var_registrar:
#         var_registrar[name] = mx.sym.Variable(name, shape=shape, init=init, dtype=np.float32)
#     return var_registrar[name]
#
# def lambda_ResMatch(data, num_filter, name, kernel=(3, 3), dilate=(1, 1), with_bn=False, dim_match=True):
#
#     lambdas = get_variable(name + '_lambda0', shape=(3,), init=mx.init.One())
#     lambdas = mx.sym.SliceChannel(lambdas, num_outputs=3, axis=0)
#     pad = dilate
#     conv0 = get_conv(name=name + '_0', data=data,
#                      num_filter=num_filter, kernel=kernel, stride=(1, 1), pad=pad, dilate=dilate, with_relu=True, with_bn=with_bn)
#     if dim_match is False:
#         data = get_conv(name=name + 'reduction', data=data, num_filter= num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), dilate=(1, 1), with_relu=False, with_bn=False)
#
#     conv1 = get_conv(name=name + '_1', data=mx.symbol.broadcast_mul(data, lambdas[1]) + conv0,
#                      num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), with_relu=True, with_bn=with_bn)
#
#     return mx.sym.broadcast_mul(data, lambdas[0]) + mx.sym.broadcast_mul(conv0, lambdas[2]) + conv1

def detect(data, output_dim, name):

    data0 = data = get_conv_bn(name='detect.0.0'+name, data=data, num_filter=32, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                       with_relu=True, bn_momentum=0.9)
    data = get_conv_bn(name='detect.0.1'+name, data=data, num_filter=32, kernel=(3, 3), stride=(1, 1), pad=(2, 2),
                       with_relu=True, bn_momentum=0.9, dilate=(2, 2))
    data = get_conv_bn(name='detect.0.2'+name, data=data, num_filter=32, kernel=(3, 3), stride=(1, 1), pad=(4, 4),
                       with_relu=True, bn_momentum=0.9, dilate=(4, 4))
    data1 = data = data + data0

    data0 = data = get_conv_bn(name='detect.1.0'+name, data=data, num_filter=64, kernel=(3, 3), stride=(2, 2), pad=(1, 1),
                       with_relu=True, bn_momentum=0.9)
    data = get_conv_bn(name='detect.1.1'+name, data=data, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(2, 2),
                       with_relu=True, bn_momentum=0.9, dilate=(2, 2))
    data = get_conv_bn(name='detect.1.2'+name, data=data, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(4, 4),
                       with_relu=True, bn_momentum=0.9, dilate=(4, 4))
    data2 = data = data0 + data

    data = get_conv_bn(name='detect.2.0'+name, data=data, num_filter=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1),
                       with_relu=True, bn_momentum=0.9)
    data = get_conv_bn(name='detect.2.1'+name, data=data, num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(2, 2),
                       with_relu=True, bn_momentum=0.9, dilate=(2, 2))

    data = get_conv_bn(name='detect.3'+name, data=data, num_filter=64, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                       with_relu=True, bn_momentum=0.9, is_conv=False)
    data = mx.sym.Concat(data2, data)
    data = get_conv_bn(name='detect.4'+name, data=data, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                       with_relu=True, bn_momentum=0.9, is_conv=True)
    data = get_conv_bn(name='detect.5'+name, data=data, num_filter=32, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                       with_relu=True, bn_momentum=0.9, is_conv=False)
    data = mx.sym.Concat(data1, data)
    data = get_conv(name='detect.6'+name, data=data, num_filter=output_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=False)
    data = mx.sym.Activation(data=data, act_type='sigmoid', name='predict_error_map'+name)

    return data
