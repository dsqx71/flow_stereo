from .symbol_util import *

def resnet(loss_scale, net_type='stereo', is_sparse = False):
    """
    resnet blocks in the place of convolution layers in Encoder of Dispnet.

    Parameters
    ----------
    loss_scale : dict of loss_scale,
        it have six loss functions which have different sizes and loss scale.
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
        Resnet symbol

    References
    ----------
        [1] KaiMing He, Deep Residual Learning for Image Recognition
    """
    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    if net_type == 'stereo':
        output_dim = 1
    elif net_type == 'flow':
        output_dim = 2

    # six loss functions with different output sizes
    labels = {'loss{}'.format(i) :  mx.sym.Variable('loss{}_label'.format(i)) for i in range(1, 7)}
    prediction = {}
    loss = []

    # siamese network, two branches share the same weights
    weights = [mx.sym.Variable('share{}_weight'.format(i)) for i in range(1,4)]
    bias    = [mx.sym.Variable('share{}_bias'.format(i)) for i in range(1,4)]
    conv1_img1, conv2_img1 = conv_share(img1, 'img1', weights, bias)
    conv1_img2, conv2_img2 = conv_share(img2, 'img2', weights, bias)

    data = mx.sym.Concat(img1, img2)

    conv0 = residual_unit(data=data, num_filter=32, stride=(1, 1), dim_match=False, name='conv0.0',
                          bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)
    conv1 = residual_unit(data=conv0, num_filter=64, stride=(2, 2), dim_match=False, name='conv1.0',
                          bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)
    conv2 = residual_unit(data=conv1, num_filter=128, stride=(2, 2), dim_match=False, name='conv2.0',
                          bottle_neck=False, bn_mom=0.9, workspace=512, memonger=False)

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
        stride = (2,2)
    elif net_type == 'flow':
        stride = (1,1)

    # Resnet block without shortcut
    conv3a = residual_unit(data=concat, num_filter=256, stride=stride, dim_match=False, name='conv3a',
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
    concat_tmp4 = mx.sym.Concat(conv2, upconv2, upsample_pr3to2)
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

    # ignore the loss functions with loss scale of zero
    keys = loss_scale.keys()
    keys.sort()
    for key in keys:
        if loss_scale[key] > 0.0:
            loss.append(get_loss(-prediction[key], labels[key], loss_scale[key], name=key,
                                 get_input=False, is_sparse = is_sparse, type=net_type))
    net = mx.sym.Group(loss)
    return net
