import mxnet as mx
import numpy as np

class SparseRegressionLoss(mx.operator.CustomOp):
    """
        SparseRegressionLoss will ignore labels with values of NaN
    """
    def __init__(self,loss_scale, is_l1):
        # due to mxnet serialization problem
        super(SparseRegressionLoss, self).__init__()
        loss_scale = float(loss_scale)
        is_l1 = bool(is_l1)
        self.loss_scale = loss_scale
        self.is_l1 = is_l1

    def forward(self, is_train, req, in_data, out_data, aux):

        x = in_data[0]
        y = out_data[0]
        self.assign(y, req[0], x)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        label = in_data[1].asnumpy()
        y = out_data[0].asnumpy()
        # find invalid labels
        mask_nan = (label != label)
        # total number of valid points
        normalize_coeff = (~mask_nan[:, 0, :, :]).sum()
        if self.is_l1:
            tmp = np.sign(y - label) * self.loss_scale / float(normalize_coeff)
        else:
            tmp = (y - label) * self.loss_scale / float(normalize_coeff)

        # ignore NaN
        tmp[mask_nan] = 0
        if normalize_coeff == 0:
            tmp[:] = 0

        self.assign(in_grad[0], req[0], mx.nd.array(tmp))

@mx.operator.register("SparseRegressionLoss")
class SparseRegressionLossProp(mx.operator.CustomOpProp):

    def __init__(self, loss_scale, is_l1):
        super(SparseRegressionLossProp, self).__init__(False)
        self.loss_scale = loss_scale
        self.is_l1 = is_l1

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]

        return [data_shape, label_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):

        return SparseRegressionLoss(self.loss_scale, self.is_l1)

def get_loss(data, label, loss_scale, name, get_input=False, is_sparse = False, type='stereo'):

    if type == 'stereo':
        data = mx.sym.Activation(data=data, act_type='relu',name=name+'relu')
    # loss
    if  is_sparse:
        loss =mx.symbol.Custom(data=data, label=label, name=name, loss_scale= loss_scale, is_l1=True,
            op_type='SparseRegressionLoss')
    else:
        loss = mx.sym.MAERegressionOutput(data=data, label=label, name=name, grad_scale=loss_scale)

    return (loss,data) if get_input else loss


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=False, bn_mom=0.9, workspace=512,
                  memonger=False, factor=0.25):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # A bit difference from origin paper
        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter*factor), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                   no_bias=False, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=1e-5 + 1e-10, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*factor), kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=False, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=1e-5 + 1e-10, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')

        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1, 1),
                                   pad=(0, 0),
                                   no_bias=False, workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, momentum=bn_mom, eps=1e-5 + 1e-10, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')

        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=False,
                                          workspace=workspace, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')

        return act3 + shortcut
    else:
        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                   no_bias=False, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=1e-5 + 1e-10, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

        conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                   no_bias=False, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=1e-5 + 1e-10, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')

        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=False,
                                          workspace=workspace, name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')

        return act2 + shortcut

def conv_share(sym, name, weights, bias):
    """
     siamese network of Dispnet and Flownet
    """
    conv1 = mx.sym.Convolution(data=sym, pad=(3, 3), kernel=(7, 7), stride=(2, 2), num_filter=64,
                               weight=weights[0], bias=bias[0], name='conv1' + name)
    conv1 = mx.sym.LeakyReLU(data = conv1,  act_type = 'leaky', slope  = 0.1)

    conv2 = mx.sym.Convolution(data = conv1, pad  = (2,2),  kernel=(5,5), stride=(2,2), num_filter=128,
                                 weight=weights[1], bias=bias[1], name='conv2' + name)
    conv2 = mx.sym.LeakyReLU(data = conv2, act_type = 'leaky', slope = 0.1)

    return conv1, conv2

def get_conv(name, data, num_filter, kernel, stride, pad, with_relu,  dilate=(1, 1)):

    conv = mx.symbol.Convolution(name=name, data=data, num_filter=num_filter, kernel=kernel,
                                 stride=stride, pad=pad, dilate=dilate, no_bias=True)

    return (mx.sym.Activation(data=conv, act_type='relu', name=name+'_relu') if with_relu else conv)

def warp(img, flow, name):
    grid = mx.sym.GridGenerator(flow, transform_type='warp',
                                name=name + '_gridgenerator')
    img_warped = mx.sym.BilinearSampler(img, grid,
                                        name=name + '_warp')
    return img_warped

def get_conv_bn(name, data, num_filter, kernel, stride, pad, with_relu, bn_momentum=0.9, dilate=(1, 1), weight=None,
             bias=None, is_conv=True):

    if is_conv is True:
        if weight is None:
            conv = mx.symbol.Convolution(
                name=name,
                data=data,
                num_filter=num_filter,
                kernel=kernel,
                stride=stride,
                pad=pad,
                dilate=dilate,
                no_bias=False,
                workspace=4096)
        else:
            conv = mx.symbol.Convolution(
                name=name,
                data=data,
                num_filter=num_filter,
                kernel=kernel,
                stride=stride,
                pad=pad,
                weight=weight,
                bias=bias,
                dilate=dilate,
                no_bias=False,
                workspace=4096)
    else:
        conv = mx.symbol.Deconvolution(
            name=name,
            data=data,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=False,
            workspace=4096)
    bn = mx.symbol.BatchNorm(
        name=name + '_bn',
        data=conv,
        fix_gamma=False,
        momentum=bn_momentum,
        eps=1e-5 + 1e-10)

    return mx.sym.LeakyReLU(data=bn, act_type='leaky', slope=0.1) if with_relu else bn

def warp_flownet(img1, img2, flow, name, factor=2, is_block=False, is_bilinear=False):

    if is_block:
        flow = mx.sym.BlockGrad(data=flow, name = name+'_block')

    if is_bilinear:
        flow = mx.sym.UpSampling(data=flow, scale=factor, num_filter=2,
                                 num_args=1, sample_type='bilinear', name='upsamplingop_flow{}'.format(name))
    else:
        flow = mx.sym.UpSampling(arg0=flow, scale=factor, num_filter=2,
                                 num_args=1, sample_type='nearest', name='upsamplingop_flow{}'.format(name))

    img2_warped = warp(img=img2, flow=flow, name='flownet-{}-warp'.format(name))
    error = mx.sym.abs(img2_warped - img1)
    data = mx.sym.Concat(img1, img2, flow, error, img2_warped)

    return data

def warp_dispnet(img1, img2, disp, name, factor=2):

    if factor > 1:
        disp = mx.sym.UpSampling(arg0=disp, scale=factor, num_filter=1,
                                 num_args=1, sample_type='nearest',
                                 name='upsamplingop_disp{}'.format(name))
    disp = mx.sym.BlockGrad(data=disp, name='blockgrad_disp{}'.format(name))
    flow = mx.sym.concat(disp, mx.sym.zeros_like(disp))
    img2_warped = warp(img=img2, flow=flow, name='dispnet-{}-warp'.format(name))
    error = mx.sym.square(img1 - img2_warped)
    error = mx.sym.sum(error, axis=1, keepdims = True)
    error = mx.sym.sqrt(error)
    data = mx.sym.Concat(img1, img2, img2_warped, disp, error)

    return data

