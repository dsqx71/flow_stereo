from .symbol_util import *
import numpy as np
var_registrar = {}
def get_variable(name, shape=None, init=None):
    global var_registrar
    if name not in var_registrar:
        var_registrar[name] = mx.sym.Variable(name, shape=shape, init=init, dtype=np.float32)
    return var_registrar[name]


def lambda_ResMatch(data, num_filter, name, kernel=(3, 3), dilate=(1, 1), with_bn=False):

    lambdas = get_variable(name + '_lambda0', shape=(3,), init=mx.init.One())
    lambdas = mx.sym.SliceChannel(lambdas, num_outputs=3, axis=0)
    pad = dilate
    conv0 = get_conv(name=name + '_0', data=data,
                     num_filter=num_filter, kernel=kernel, stride=(1, 1), pad=pad, dilate=dilate, with_relu=True, with_bn=with_bn)
    conv1 = get_conv(name=name + '_1', data=mx.symbol.broadcast_mul(data, lambdas[1]) + conv0,
                     num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), with_relu=True, with_bn=with_bn)

    return mx.sym.broadcast_mul(data, lambdas[0]) + mx.sym.broadcast_mul(conv0, lambdas[2]) + conv1


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

def conv_share(sym):

    # Encoder
    # Level 0
    conv0 = get_conv(name='conv0_16.0', data=sym, num_filter= 13, kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), with_relu=True, with_bn=True)
    conv0 = mx.sym.Concat(conv0, sym)

    # Level 1 - 64
    conv1 = get_conv(name='conv1_64', data=conv0, num_filter=64, kernel=(3, 3), stride=(2, 2), pad=(1, 1), with_relu=True, with_bn=True)
    conv1 = lambda_ResMatch(data = conv1, num_filter = 64, name= 'conv1_64.0_lambda_ResMatch', with_bn=True, dilate=(1, 1))

    # Level 2 - 128
    conv2 = get_conv(name='conv2_128', data=conv1, num_filter=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), with_relu=True, with_bn=True)
    conv2 = lambda_ResMatch(data = conv2, num_filter = 128, name= 'conv2_128.0_lambda_ResMatch', with_bn=True, dilate=(1, 1))

    # Level 3 - 256
    conv3 = get_conv(name='conv3_256', data=conv2, num_filter=256, kernel=(3, 3), stride=(2, 2), pad=(1, 1), with_relu=True, with_bn=True)
    conv3 = lambda_ResMatch(data = conv3, num_filter = 256, name= 'conv3_256.0_lambda_ResMatch', with_bn=True, dilate=(1, 1))

    # Decoder
    # Level 2 - 128
    deconv2 = get_conv(name='deconv2_128', data=conv3, num_filter=128, kernel=(4, 4), stride=(2, 2), pad=(1, 1), with_relu=True, with_bn=True, is_conv=False)
    conv2 = mx.sym.concat(deconv2, conv2)
    conv2 = get_conv(name='conv2', data=conv2, num_filter=128,  kernel=(3, 3), stride=(1, 1), pad=(1, 1), with_relu=True, with_bn=True)
    # Level 1 - 64
    deconv1 = get_conv(name='deconv1_64', data=conv2, num_filter=64,  kernel=(4, 4), stride=(2, 2), pad=(1, 1), with_relu=True, with_bn=True, is_conv=False)
    conv1 = mx.sym.concat(deconv1, conv1)
    conv1 = get_conv(name='conv1', data=conv1, num_filter=64,  kernel=(3, 3), stride=(1, 1), pad=(1, 1), with_relu=True, with_bn=True)

    return conv1

def dispnet_c(img1, img2, labels, loss_scale, net_type='flow', is_sparse=False):

    if net_type == 'stereo':
        output_dim = 1
    elif net_type == 'flow':
        output_dim = 2

    prediction = {}
    loss = []

    # siamese network, two branches share the same weights
    conv1_img1 = conv_share(img1)
    conv1_img2 = conv_share(img2)

    data = mx.sym.concat(img1, img2)
    data0 = conv1 =  get_conv(name='Convolution1.0', data=data, num_filter=64, kernel=(5, 5), stride=(2, 2), pad=(2, 2), with_relu=True)
    conv1 =  get_conv(name='Convolution1.1', data=conv1, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), with_relu=True)
    conv1 = conv1 + data0

    # difference between DispNet and dispnet
    # if net_type == 'stereo':
    corr = mx.sym.Correlation1D(data1=conv1_img1, data2=conv1_img2, pad_size=80, kernel_size=1,
                                max_displacement=80, stride1=1, stride2=1, single_side=0)
    disparity = mx.sym.arange(80, stop=-81, step=-1, repeat=1) * 2
    corr = mx.symbol.softmax(data=corr, axis=1, name='corr_softmax')
    tmp = mx.sym.transpose(corr, axes=(0, 2, 3 ,1))
    soft_argmax = mx.sym.broadcast_mul(tmp, disparity)
    soft_argmax = mx.sym.sum(soft_argmax, axis=3, keepdims=True)
    soft_argmax = mx.sym.transpose(soft_argmax, axes=(0, 3, 1, 2))
    # soft_argmax = mx.sym.UpSampling(arg0=soft_argmax, scale=4, num_filter=1,
    #                                 num_args=1, sample_type='nearest', name='upsamplingop')
    prediction['loss0'] = soft_argmax
    # labels['loss0'] = mx.sym.where(condition=labels['loss0']>200, x = prediction['loss0'], y = labels['loss0'], name='where')
    concat = mx.sym.Concat(corr, conv1, soft_argmax)
    #
    conv2 = mx.sym.Convolution(concat, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=128, name='Convolution2')
    conv2 = mx.sym.LeakyReLU(data=conv2, act_type='leaky', slope=0.1)

    conv3a = mx.sym.Convolution(conv2, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=256, name='Convolution3a')
    conv3a = mx.sym.LeakyReLU(data=conv3a, act_type='leaky', slope=0.1)

    conv3b = mx.sym.Convolution(conv3a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256, name='conv3b')
    conv3b = mx.sym.LeakyReLU(data=conv3b, act_type='leaky', slope=0.1)

    conv4a = mx.sym.Convolution(conv3b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=256, name='conv4a')
    conv4a = mx.sym.LeakyReLU(data=conv4a, act_type='leaky', slope=0.1)

    conv4b = mx.sym.Convolution(conv4a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256, name='conv4b')
    conv4b = mx.sym.LeakyReLU(data=conv4b, act_type='leaky', slope=0.1)

    conv5a = mx.sym.Convolution(conv4b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=512, name='conv5a')
    conv5a = mx.sym.LeakyReLU(data=conv5a, act_type='leaky', slope=0.1)

    conv5b = mx.sym.Convolution(conv5a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name='conv5b')
    conv5b = mx.sym.LeakyReLU(data=conv5b, act_type='leaky', slope=0.1)

    conv6a = mx.sym.Convolution(conv5b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=512, name='conv6a')
    conv6a = mx.sym.LeakyReLU(data=conv6a, act_type='leaky', slope=0.1)

    conv6b = mx.sym.Convolution(conv6a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name='conv6b')
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
    iconv4 = mx.sym.Convolution(concat_tmp2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256, name='iconv4')
    pr4 = mx.sym.Convolution(iconv4, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr4')
    prediction['loss4'] = pr4

    upconv3 = mx.sym.Deconvolution(iconv4, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=128, name='upconv3',
                                   no_bias=True)
    upconv3 = mx.sym.LeakyReLU(data=upconv3, act_type='leaky', slope=0.1)

    upsample_pr4to3 = mx.sym.Deconvolution(pr4, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_pr4to3', no_bias=True)
    concat_tmp3 = mx.sym.Concat(conv3b, upconv3, upsample_pr4to3)
    iconv3 = mx.sym.Convolution(concat_tmp3, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=128, name='iconv3')
    pr3 = mx.sym.Convolution(iconv3, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr3')
    prediction['loss3'] = pr3

    upconv2 = mx.sym.Deconvolution(iconv3, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=64, name='upconv2',
                                   no_bias=True)
    upconv2 = mx.sym.LeakyReLU(data=upconv2, act_type='leaky', slope=0.1)

    upsample_pr3to2 = mx.sym.Deconvolution(pr3, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_pr3to2', no_bias=True)
    concat_tmp4 = mx.sym.Concat(conv2, upconv2, upsample_pr3to2)
    iconv2 = mx.sym.Convolution(concat_tmp4, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=64, name='iconv2')
    pr2 = mx.sym.Convolution(iconv2, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr2')
    prediction['loss2'] = pr2

    upconv1 = mx.sym.Deconvolution(iconv2, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=32, name='upconv1',
                                   no_bias=True)
    upconv1 = mx.sym.LeakyReLU(data=upconv1, act_type='leaky', slope=0.1)
    upsample_pr2to1 = mx.sym.Deconvolution(pr2, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=output_dim,
                                           name='upsample_pr2to1', no_bias=True)
    concat_tmp5 = mx.sym.Concat(conv1, upconv1, upsample_pr2to1)
    iconv1 = mx.sym.Convolution(concat_tmp5, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=32, name='iconv1')
    pr1 = mx.sym.Convolution(iconv1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr1')
    prediction['loss1'] = pr1

    # ignore the loss functions with loss scale of zero
    keys = loss_scale.keys()
    keys.sort(reverse = True)
    for key in keys:
        if loss_scale[key] > 0.0:
            loss.append(get_loss(prediction[key], labels[key], loss_scale[key], name=key,
                                 get_input=False, is_sparse=is_sparse, type=net_type))

    loss.append(mx.sym.BlockGrad(data=soft_argmax,name='soft_argmax'))
    loss.append(mx.sym.BlockGrad(data=conv1_img1, name='conv1_block'))
    loss.append(mx.sym.BlockGrad(data=conv1_img2, name='conv2_block'))
    loss.append(mx.sym.BlockGrad(data=corr, name='corr_block'))
    return prediction, loss


def dispnet2CSS(loss_scale, net_type='flow', is_sparse=False):

    # input
    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    # six loss functions with different output sizes
    labels = {'loss{}'.format(i): mx.sym.Variable('loss{}_label'.format(i)) for i in range(0, 7)}

    # dispnet-c
    dispnetc_prediction, dispnetc_loss = dispnet_c(img1, img2, labels, loss_scale['dispnetc'],
                                                   net_type=net_type, is_sparse=is_sparse)
    dispnetc_params = dispnetc_loss[0].list_arguments()

    loss = dispnetc_loss
    keys = list(var_registrar.keys())
    keys.sort()
    # for key in keys:
    #     if 'lambda0' in key:
    #         tmp = mx.sym.BlockGrad(data=var_registrar[key], name=key+'_block')
    #         loss.append(tmp)
    net = mx.sym.Group(loss)

    return net, dispnetc_params
