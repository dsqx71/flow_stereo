from symbol_util import *

def Gfunction(img1, img2_warped, img3, name, flow = None, factor=1, out_dim=2):

    data = mx.sym.Concat(img1, img2_warped, img3, flow) if flow is not None else mx.sym.Concat(img1, img2_warped)

    data = get_conv(name + '.0', data, num_filter= 32 * factor,
                    kernel = (7, 7), stride= (1, 1),
                    pad=(3, 3), with_relu=True, dilate=(1, 1))

    data = get_conv(name + '.1', data, num_filter= 64 * factor,
                    kernel = (7, 7), stride= (1, 1),
                    pad=(3, 3), with_relu=True, dilate=(1, 1))

    data = get_conv(name + '.2', data, num_filter= 32 * factor,
                    kernel = (7, 7), stride= (1, 1),
                    pad=(3, 3), with_relu=True, dilate=(1, 1))

    data = get_conv(name + '.3', data, num_filter= 16 * factor,
                    kernel = (7, 7), stride= (1, 1),
                    pad=(3, 3), with_relu=True, dilate=(1, 1))

    data = get_conv(name + '.4', data, num_filter=16 * factor,
                    kernel=(7, 7), stride=(1, 1),
                    pad=(3, 3), with_relu=True, dilate=(1, 1))

    data = mx.sym.Convolution(data,  kernel = (7, 7), stride=(1, 1), pad=(3, 3),
                              num_filter=out_dim, name=name+'.predictor')

    return data

def block(img1, img2, flow, name, factor=1, out_dim=2, up_sample=True):

    if up_sample:
        flow = mx.sym.BlockGrad(data=flow, name=name+'_blockflow')
        flow = mx.sym.UpSampling(data=flow, scale=2, num_filter=2,
                                 num_args=1, sample_type='bilinear', name='upsamplingop_flow{}'.format(name))
    img2_warped = warp(img2, flow, name=name+'warp')
    res = Gfunction(img1, img2_warped, img2, name, flow, factor, out_dim=out_dim)
    data = res + flow

    return data

def init(img1, img2, out_dim=2):

    v0 = Gfunction(img1=img1, img2_warped=img2, img3=None, name='v0', flow=None, factor=1, out_dim=out_dim)

    return v0

def spynet(loss_scale, net_type='stereo', is_sparse = False):
    """
    create Spynet symbol.

    Note:
        This architecture has slight difference from original one of spynet

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
        spynet symbol

    References
    ----------
        [1] Optical Flow Estimation using a Spatial Pyramid Network

    """
    out_dim = 2 if net_type == 'flow' else 1

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    labels = {'loss{}'.format(i): mx.sym.Variable('loss{}_label'.format(i)) for i in range(1, 6)}
    prediction = {}
    loss = []
    img1_downsample = []
    img2_downsample = []

    for i in range(1, 5):
        img1_downsample.append(mx.sym.Pooling(data=img1, kernel=(2**i, 2**i), stride=(2**i, 2**i), pool_type='avg'))
        img2_downsample.append(mx.sym.Pooling(data=img2, kernel=(2**i, 2**i), stride=(2**i, 2**i), pool_type='avg'))


    prediction['loss5'] = Gfunction(img1=img1_downsample[3], img2_warped=img2_downsample[3],
                                    img3=None, name='v0', flow=None, factor=1, out_dim=out_dim)
    prediction['loss4'] = block(img1_downsample[2], img2_downsample[2],
                                prediction['loss5'] * 2, name = 'v1', factor=1, out_dim=out_dim)
    prediction['loss3'] = block(img1_downsample[1], img2_downsample[1],
                                prediction['loss4'] * 2, name = 'v2', factor=1, out_dim=out_dim)
    prediction['loss2'] = block(img1_downsample[0], img2_downsample[0],
                                prediction['loss3'] * 2, name = 'v3', factor=1, out_dim=out_dim)
    prediction['loss1'] = block(img1, img2,
                                prediction['loss2'] * 2, name = 'v4', factor=1, out_dim=out_dim)

    # Loss
    keys = loss_scale.keys()
    keys.sort()
    discount = 1.0
    for key in keys:
        if loss_scale[key] > 0.0:
            loss.append(get_loss(prediction[key], labels[key] / discount, loss_scale[key], name=key,
                                 get_input=False, is_sparse=is_sparse, type=net_type))
        discount *= 2.0
    net = mx.sym.Group(loss)

    return net







