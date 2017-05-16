from .symbol_util import *

def flownets_half(loss_scale, net_type='stereo', is_sparse = False):

    name = 'flownets_half'

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    if net_type == 'stereo':
        output_dim = 1
    elif net_type == 'flow':
        output_dim = 2

    # six loss functions with different output sizes
    labels = {'loss{}'.format(i): mx.sym.Variable('loss{}_label'.format(i)) for i in range(1, 7)}
    prediction = {}
    loss = []

    data = mx.sym.Concat(img1, img2)

    # The structure below is similar to VGG
    conv1 = mx.sym.Convolution(data, pad=(3, 3), kernel=(7, 7), stride=(2, 2), num_filter=32, name=name+'conv1')
    conv1 = mx.sym.LeakyReLU(data=conv1, act_type='leaky', slope=0.1)

    conv2 = mx.sym.Convolution(conv1, pad=(2, 2), kernel=(5, 5), stride=(2, 2), num_filter=64, name=name+'conv2')
    conv2 = mx.sym.LeakyReLU(data=conv2, act_type='leaky', slope=0.1)

    conv3a = mx.sym.Convolution(conv2, pad=(2, 2), kernel=(5, 5), stride=(2, 2), num_filter=128, name=name+'conv3a')
    conv3a = mx.sym.LeakyReLU(data=conv3a, act_type='leaky', slope=0.1)

    conv3b = mx.sym.Convolution(conv3a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=128, name=name+'conv3b')
    conv3b = mx.sym.LeakyReLU(data=conv3b, act_type='leaky', slope=0.1)

    conv4a = mx.sym.Convolution(conv3b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=256, name=name+'conv4a')
    conv4a = mx.sym.LeakyReLU(data=conv4a, act_type='leaky', slope=0.1)

    conv4b = mx.sym.Convolution(conv4a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256, name=name+'conv4b')
    conv4b = mx.sym.LeakyReLU(data=conv4b, act_type='leaky', slope=0.1)

    conv5a = mx.sym.Convolution(conv4b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=256, name=name+'conv5a')
    conv5a = mx.sym.LeakyReLU(data=conv5a, act_type='leaky', slope=0.1)

    conv5b = mx.sym.Convolution(conv5a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=256, name=name+'conv5b')
    conv5b = mx.sym.LeakyReLU(data=conv5b, act_type='leaky', slope=0.1)

    conv6a = mx.sym.Convolution(conv5b, pad=(1, 1), kernel=(3, 3), stride=(2, 2), num_filter=512, name=name+'conv6a')
    conv6a = mx.sym.LeakyReLU(data=conv6a, act_type='leaky', slope=0.1)

    conv6b = mx.sym.Convolution(conv6a, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=512, name=name+'conv6b')
    conv6b = mx.sym.LeakyReLU(data=conv6b, act_type='leaky', slope=0.1, )

    pr6 = mx.sym.Convolution(conv6b,pad= (1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name=name+'pr6')
    prediction['loss6'] = pr6

    upsample_pr6to5 = mx.sym.Deconvolution(pr6, pad=(1,1), kernel=(4,4), stride=(2,2), num_filter=output_dim,
                                           name=name+'upsample_pr6to5',no_bias=True)
    upconv5 = mx.sym.Deconvolution(conv6b,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=512,name=name+'upconv5',no_bias=True)
    upconv5 = mx.sym.LeakyReLU(data = upconv5,act_type = 'leaky',slope  = 0.1)
    concat_tmp = mx.sym.Concat(conv5b,upconv5,upsample_pr6to5,dim=1)

    iconv5 = mx.sym.Convolution(concat_tmp,pad = (1,1),kernel=(3,3),stride=(1,1),num_filter = 512,name=name+'iconv5')

    pr5  = mx.sym.Convolution(iconv5, pad = (1,1),kernel=(3,3),stride=(1,1),num_filter = output_dim,name=name+'pr5')
    prediction['loss5'] = pr5

    upconv4 = mx.sym.Deconvolution(iconv5,pad = (1,1),kernel= (4,4),stride = (2,2),num_filter=256,name=name+'upconv4',no_bias=True)
    upconv4 = mx.sym.LeakyReLU(data = upconv4,act_type = 'leaky',slope  = 0.1 )

    upsample_pr5to4 = mx.sym.Deconvolution(pr5,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name=name+'upsample_pr5to4',no_bias=True)

    concat_tmp2 = mx.sym.Concat(conv4b,upconv4,upsample_pr5to4)
    iconv4  = mx.sym.Convolution(concat_tmp2,pad = (1,1),kernel = (3,3),stride=(1,1),num_filter=256,name=name+'iconv4')
    pr4 = mx.sym.Convolution(iconv4,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name=name+'pr4')
    prediction['loss4'] = pr4

    upconv3 = mx.sym.Deconvolution(iconv4,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=128,name=name+'upconv3',no_bias=True)
    upconv3 = mx.sym.LeakyReLU(data = upconv3,act_type = 'leaky',slope  = 0.1 )

    upsample_pr4to3 = mx.sym.Deconvolution(pr4,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name=name+'upsample_pr4to3',no_bias=True)
    concat_tmp3 = mx.sym.Concat(conv3b,upconv3,upsample_pr4to3)
    iconv3 = mx.sym.Convolution(concat_tmp3,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter = 128,name=name+'iconv3')
    pr3 = mx.sym.Convolution(iconv3,pad = (1,1), kernel = (3,3), stride = (1,1),num_filter = output_dim,name=name+'pr3')
    prediction['loss3'] = pr3

    upconv2 = mx.sym.Deconvolution(iconv3,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=64,name=name+'upconv2',no_bias=True)
    upconv2 = mx.sym.LeakyReLU(data = upconv2,act_type = 'leaky',slope  = 0.1  )

    upsample_pr3to2 = mx.sym.Deconvolution(pr3,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name=name+'upsample_pr3to2',no_bias=True)
    concat_tmp4 = mx.sym.Concat(conv2, upconv2, upsample_pr3to2)
    iconv2 = mx.sym.Convolution(concat_tmp4,pad = (1,1),kernel = (3,3),stride= (1,1),num_filter = 64,name=name+'iconv2')
    pr2 = mx.sym.Convolution(iconv2,pad = (1,1),kernel=(3,3),stride = (1,1),num_filter = output_dim,name=name+'pr2')
    prediction['loss2'] = pr2

    upconv1 = mx.sym.Deconvolution(iconv2,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter = 32,name=name+'upconv1',no_bias=True)
    upconv1 = mx.sym.LeakyReLU(data = upconv1,act_type = 'leaky',slope  = 0.1 )
    upsample_pr2to1 = mx.sym.Deconvolution(pr2,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=output_dim,name=name+'upsample_pr2to1',no_bias=True)
    concat_tmp5 = mx.sym.Concat(conv1, upconv1, upsample_pr2to1)
    iconv1 = mx.sym.Convolution(concat_tmp5,pad=(1,1),kernel = (3,3),stride=(1,1),num_filter=32,name=name+'iconv1')
    pr1 = mx.sym.Convolution(iconv1, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim,name=name+'pr1')
    prediction['loss1'] = pr1

    # ignore the loss functions with loss scale of zero
    keys = loss_scale.keys()
    keys.sort()
    for key in keys:
        loss.append(get_loss(prediction[key], labels[key], loss_scale[key], name=key+name,
                             get_input=False, is_sparse = is_sparse, type=net_type))

    net = mx.sym.Group(loss)
    return net