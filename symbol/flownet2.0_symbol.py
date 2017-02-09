import mxnet as mx
import numpy as np
from config import cfg
from symbol.drr_symbol import detect_replace_refine

class SparseRegressionLoss(mx.operator.NumpyOp):
    '''
        if label is nan, don't compute gradient
    '''

    def __init__(self,is_l1,loss_scale):

        super(SparseRegressionLoss, self).__init__(False)
        self.is_L1 = is_l1
        self.loss_scale = loss_scale

    def list_arguments(self):

        return ['data', 'label']

    def list_outputs(self):

        return ['output']

    def infer_shape(self, in_shape):

        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]

        return [data_shape, label_shape], [output_shape]

    def forward(self,in_data,out_data):

        x = in_data[0]
        y = out_data[0]
        y[:] = x

    def backward(self,out_grad, in_data, out_data, in_grad):

        label = in_data[1]
        y = out_data[0]
        mask = (label!=label)
        label[mask] = y[mask]
        # mask = (label>0) & (label<30)
        # mask = np.abs(y-label) < 0.3
        # label[mask] = y[mask]
        # mask = (np.abs(y-label) >= 1) & (np.abs(y-label)/np.abs(label)>=0.02)
        dx = in_grad[0]
        if self.is_L1:
            dx[:] = np.sign(y-label)*self.loss_scale
        else:
            dx[:] = (y-label)*self.loss_scale


def conv_unit(sym, name, weights, bias):

    conv1 = mx.sym.Convolution(data=sym,pad=(3, 3), kernel=(7, 7),stride=(2, 2),num_filter=64,
                               weight=weights[0], bias=bias[0], name='conv1' + name)
    conv1 = mx.sym.LeakyReLU(data = conv1,  act_type = 'leaky', slope  = 0.1 )

    conv2 = mx.sym.Convolution(data = conv1, pad  = (2,2),  kernel=(5,5),stride=(2,2),num_filter=128,
                                 weight = weights[1], bias = bias[1], name='conv2' + name)
    conv2 = mx.sym.LeakyReLU(data = conv2, act_type = 'leaky', slope = 0.1)

    return conv1,conv2,

def FlowNetC(net_type='stereo', is_sparse = False):

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')
    if net_type == 'stereo':
        output_dim = 1
    else:
        output_dim = 2

    downsample0 = mx.sym.Variable(net_type + '_downsample1')
    downsample1 = mx.sym.Variable(net_type + '_downsample2')
    downsample2 = mx.sym.Variable(net_type + '_downsample3')
    downsample3 = mx.sym.Variable(net_type + '_downsample4')
    downsample4 = mx.sym.Variable(net_type + '_downsample5')
    downsample5 = mx.sym.Variable(net_type + '_downsample6')
    downsample6 = mx.sym.Variable(net_type + '_downsample7')

    weights = [mx.sym.Variable('share{}_weight'.format(i)) for i in range(1,4)]
    bias    = [mx.sym.Variable('share{}_bias'.format(i)) for i in range(1,4)]

    conv1_img1, conv2_img1 = conv_unit(img1, 'img1', weights, bias)
    conv1_img2, conv2_img2 = conv_unit(img2, 'img2', weights, bias)

    if net_type =='stereo':
        corr = mx.sym.Correlation1D(data1=conv2_img1, data2=conv2_img2, pad_size=40, kernel_size=1,
                                    max_displacement=40, stride1=1, stride2=1)

        conv_redir = mx.sym.Convolution(data=conv2_img1, pad=(0, 0), kernel=(1, 1), stride=(1, 1), num_filter=64,
                                        name='conv_redir')

        conv_redir = mx.sym.LeakyReLU(data=conv_redir, act_type='leaky', slope=0.1)
        concat = mx.sym.Concat(corr, conv_redir)
    else:
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
    else:
        stride = (1,1)
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

    upsample_pr6to5 = mx.sym.Deconvolution(pr6, pad=(1,1), kernel=(4,4), stride=(2,2), num_filter=1,
                                           name='upsample_pr6to5',no_bias=True)
    upconv5 = mx.sym.Deconvolution(conv6b,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=512,name='upconv5',no_bias=True)
    upconv5 = mx.sym.LeakyReLU(data = upconv5,act_type = 'leaky',slope  = 0.1)
    concat_tmp = mx.sym.Concat(conv5b,upconv5,upsample_pr6to5,dim=1)

    iconv5 = mx.sym.Convolution(concat_tmp,pad = (1,1),kernel=(3,3),stride=(1,1),num_filter = 512,name='iconv5')

    pr5  = mx.sym.Convolution(iconv5, pad = (1,1),kernel=(3,3),stride=(1,1),num_filter = output_dim,name='pr5')

    upconv4 = mx.sym.Deconvolution(iconv5,pad = (1,1),kernel= (4,4),stride = (2,2),num_filter=256,name='upconv4',no_bias=True)
    upconv4 = mx.sym.LeakyReLU(data = upconv4,act_type = 'leaky',slope  = 0.1 )

    upsample_pr5to4 = mx.sym.Deconvolution(pr5,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=1,name='upsample_pr5to4',no_bias=True)

    concat_tmp2 = mx.sym.Concat(conv4b,upconv4,upsample_pr5to4)
    iconv4  = mx.sym.Convolution(concat_tmp2,pad = (1,1),kernel = (3,3),stride=(1,1),num_filter=256,name='iconv4')
    pr4 = mx.sym.Convolution(iconv4,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name='pr4')

    upconv3 = mx.sym.Deconvolution(iconv4,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=128,name='upconv3',no_bias=True)
    upconv3 = mx.sym.LeakyReLU(data = upconv3,act_type = 'leaky',slope  = 0.1 )

    upsample_pr4to3 = mx.sym.Deconvolution(pr4,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=1,name='upsample_pr4to3',no_bias=True)
    concat_tmp3 = mx.sym.Concat(conv3b,upconv3,upsample_pr4to3)
    iconv3 = mx.sym.Convolution(concat_tmp3,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter = 128,name='iconv3')
    pr3 = mx.sym.Convolution(iconv3,pad = (1,1), kernel = (3,3), stride = (1,1),num_filter = output_dim,name='pr3')


    upconv2 = mx.sym.Deconvolution(iconv3,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=64,name='upconv2',no_bias=True)
    upconv2 = mx.sym.LeakyReLU(data = upconv2,act_type = 'leaky',slope  = 0.1  )

    upsample_pr3to2 = mx.sym.Deconvolution(pr3,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=1,name='upsample_pr3to2',no_bias=True)

    concat_tmp4 = mx.sym.Concat(conv2_img1,upconv2,upsample_pr3to2)

    iconv2 = mx.sym.Convolution(concat_tmp4,pad = (1,1),kernel = (3,3),stride= (1,1),num_filter = 64,name='iconv2')
    pr2 = mx.sym.Convolution(iconv2,pad = (1,1),kernel=(3,3),stride = (1,1),num_filter = output_dim,name='pr2')

    upconv1 = mx.sym.Deconvolution(iconv2,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter = 32,name='upconv1',no_bias=True)
    upconv1 = mx.sym.LeakyReLU(data = upconv1,act_type = 'leaky',slope  = 0.1 )

    upsample_pr2to1 = mx.sym.Deconvolution(pr2,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=1,name='upsample_pr2to1',no_bias=True)

    concat_tmp5 = mx.sym.Concat(conv1_img1,upconv1,upsample_pr2to1)
    iconv1 = mx.sym.Convolution(concat_tmp5,pad=(1,1),kernel = (3,3),stride=(1,1),num_filter=32,name='iconv1')
    pr1 = mx.sym.Convolution(iconv1,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name='pr1')

    # drr
    pr = mx.sym.Deconvolution(pr1, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=1, name='init_label.drr',
                              no_bias=True)
    data = mx.sym.Concat(img1, img2)
    pr_final = detect_replace_refine(init_label=pr, input_feature=data, output_dim=output_dim, name='drr')

    loss = get_loss(pr_final, downsample0, 1.0, name='loss', get_data=False, is_sparse=is_sparse, type=net_type)
    loss0 = get_loss(pr, downsample0, 0.10, name='loss0', get_data=False, is_sparse = is_sparse,type=net_type)
    loss1 = get_loss(pr1, downsample1,  0.0, name='loss1', get_data=False, is_sparse = is_sparse,type=net_type)
    # loss2 = get_loss(pr2, downsample2, loss2_scale, name='loss2', get_data=False, is_sparse = is_sparse,type=net_type)
    # loss3 = get_loss(pr3, downsample3, loss3_scale, name='loss3', get_data=False, is_sparse = is_sparse,type=net_type)
    # loss4 = get_loss(pr4, downsample4, loss4_scale, name='loss4', get_data=False, is_sparse = is_sparse,type=net_type)
    # loss5 = get_loss(pr5, downsample5, loss5_scale, name='loss5', get_data=False, is_sparse = is_sparse,type=net_type)
    # loss6 = get_loss(pr6, downsample6, loss6_scale, name='loss6', get_data=False, is_sparse = is_sparse,type=net_type)

    net = mx.sym.Group([loss])

    return net

def get_loss(data,label,grad_scale,name,get_data=False, is_sparse = False, type='stereo'):

    if type == 'stereo':
        data = mx.sym.Activation(data=data, act_type='relu',name=name+'relu')

    if  is_sparse:
        loss = SparseRegressionLoss(is_l1=False, loss_scale=grad_scale)
        loss = loss(data=data, label=label)
    else:
        loss = mx.sym.MAERegressionOutput(data=data, label=label, name=name, grad_scale=grad_scale)

    return (loss,data) if get_data else loss
