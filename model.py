import mxnet as mx
import numpy as np
from config import cfg

class SparseRegressionLoss(mx.operator.NumpyOp):

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

    def backward(self,out_grad,in_data,out_data,in_grad):

        label = in_data[1]
        mask = (label > 0).astype(int)
        mask2 = (label)
        y = out_data[0]
        dx = in_grad[0]
        if self.is_L1:
            dx[:] = np.sign(y-label)*mask*self.loss_scale
        else:
            dx[:] = (y-label)*mask*self.loss_scale

def flow_and_stereo_net(net_type,is_sparse,is_l1,
                        loss0_scale=cfg.MODEL.loss0_scale,
                        loss1_scale=cfg.MODEL.loss1_scale, loss2_scale=cfg.MODEL.loss2_scale,
                        loss3_scale=cfg.MODEL.loss3_scale, loss4_scale=cfg.MODEL.loss4_scale,
                        loss5_scale=cfg.MODEL.loss5_scale, loss6_scale=cfg.MODEL.loss6_scale
                        ):
    """
        Dispnet: A large Dataset to train Convolutional networks for disparity, optical flow,and scene flow estimation

        The  architectures of dispnet and flownet are the same

        loss_scale : the weight of loss layer
    """

    if net_type == 'flow':
        output_dim = 2
    elif net_type == 'stereo':
        output_dim = 1

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    # labels with different shape
    downsample0 = mx.sym.Variable(net_type + '_downsample0')
    downsample1 = mx.sym.Variable(net_type + '_downsample1')
    downsample2 = mx.sym.Variable(net_type + '_downsample2')
    downsample3 = mx.sym.Variable(net_type + '_downsample3')
    downsample4 = mx.sym.Variable(net_type + '_downsample4')
    downsample5 = mx.sym.Variable(net_type + '_downsample5')
    downsample6 = mx.sym.Variable(net_type + '_downsample6')

    concat1 = mx.sym.Concat(img1, img2, dim=1, name='concat_image')

    conv1 = mx.sym.Convolution(concat1, pad=(3, 3), kernel=(7, 7), stride=(2, 2), num_filter=64, name='conv1')
    conv1 = mx.sym.LeakyReLU(data=conv1, act_type='leaky', slope=0.1)

    conv2 = mx.sym.Convolution(conv1, pad=(2, 2), kernel=(5, 5), stride=(2, 2), num_filter=128, name='conv2')
    conv2 = mx.sym.LeakyReLU(data=conv2, act_type='leaky', slope=0.1)

    conv3a = mx.sym.Convolution(conv2, pad=(2, 2), kernel=(5, 5), stride=(2, 2), num_filter=256, name='conv3a')
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

    if is_sparse:
        loss = SparseRegressionLoss(is_l1=is_l1,loss_scale=loss6_scale)
        loss6 = loss(data=pr6,label = downsample6)
    else:
        if is_l1:
            loss6 = mx.sym.MAERegressionOutput(data = pr6,label = downsample6,grad_scale=loss6_scale,name='loss6')
        else:
            loss6 = mx.sym.LinearRegressionOutput(data = pr6,label = downsample6,grad_scale=loss6_scale,name='loss6')

    upsample_pr6to5 = mx.sym.Deconvolution(pr6,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=2,name='upsample_pr6to5',no_bias=False)
    upconv5 = mx.sym.Deconvolution(conv6b,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=512,name='upconv5',no_bias=False)
    upconv5 = mx.sym.LeakyReLU(data = upconv5,act_type = 'leaky',slope  = 0.1  )
    concat_tmp = mx.sym.Concat(upconv5,upsample_pr6to5,conv5b,dim=1)

    iconv5 = mx.sym.Convolution(concat_tmp,pad = (1,1),kernel=(3,3),stride=(1,1),num_filter = 512,name='iconv5')
    iconv5 = mx.sym.LeakyReLU(data = iconv5,act_type = 'leaky',slope  = 0.1 )

    pr5    = mx.sym.Convolution(iconv5, pad = (1,1),kernel=(3,3),stride=(1,1),num_filter = output_dim,name='pr5')
    if is_sparse:
        loss = SparseRegressionLoss(is_l1=is_l1,loss_scale=loss5_scale)
        loss5 = loss(data=pr5,label = downsample5)
    else:
        if is_l1:
            loss5 = mx.sym.MAERegressionOutput(data = pr5,label = downsample5,grad_scale=loss5_scale,name='loss5')
        else:
            loss5 = mx.sym.LinearRegressionOutput(data = pr5,label = downsample5,grad_scale=loss5_scale,name='loss5')

    upconv4 = mx.sym.Deconvolution(iconv5,pad = (1,1),kernel= (4,4),stride = (2,2),num_filter=256,name='upconv4',no_bias=False)
    upconv4 = mx.sym.LeakyReLU(data = upconv4,act_type = 'leaky',slope  = 0.1 )

    upsample_pr5to4 = mx.sym.Deconvolution(pr5,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=2,name='upsample_pr5to4',no_bias=False)

    concat_tmp2 = mx.sym.Concat(upsample_pr5to4,upconv4,conv4b)
    iconv4  = mx.sym.Convolution(concat_tmp2,pad = (1,1),kernel = (3,3),stride=(1,1),num_filter=256,name='iconv4')
    iconv4 =  mx.sym.LeakyReLU(data = iconv4,act_type = 'leaky',slope  = 0.1  )

    pr4 = mx.sym.Convolution(iconv4,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name='pr4')
    if is_sparse:
        loss = SparseRegressionLoss(is_l1=is_l1,loss_scale=loss4_scale)
        loss4 = loss(data=pr4,label = downsample4)
    else:
        if is_l1:
            loss4 = mx.sym.MAERegressionOutput(data = pr4,label = downsample4,grad_scale=loss4_scale,name='loss4')
        else:
            loss4 = mx.sym.LinearRegressionOutput(data = pr4,label = downsample4,grad_scale=loss4_scale,name='loss4')

    upconv3 = mx.sym.Deconvolution(iconv4,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=128,name='upconv3',no_bias=False)
    upconv3 = mx.sym.LeakyReLU(data = upconv3,act_type = 'leaky',slope  = 0.1 )

    upsample_pr4to3 = mx.sym.Deconvolution(pr4,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=2,name='upsample_pr4to3',no_bias=False)
    concat_tmp3 = mx.sym.Concat(upsample_pr4to3,upconv3,conv3b)
    iconv3 = mx.sym.Convolution(concat_tmp3,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter = 128,name='iconv3')
    iconv3 = mx.sym.LeakyReLU(data = iconv3,act_type = 'leaky',slope  = 0.1 )

    pr3 = mx.sym.Convolution(iconv3,pad = (1,1), kernel = (3,3), stride = (1,1),num_filter = output_dim,name='pr3')
    if is_sparse:
        loss = SparseRegressionLoss(is_l1=is_l1,loss_scale=loss3_scale)
        loss3 = loss(data=pr3,label = downsample3)
    else:
        if is_l1:
            loss3 = mx.sym.MAERegressionOutput(data = pr3,label = downsample3,grad_scale=loss3_scale,name='loss3')
        else:
            loss3 = mx.sym.LinearRegressionOutput(data = pr3,label = downsample3,grad_scale=loss3_scale,name='loss3')

    upconv2 = mx.sym.Deconvolution(iconv3,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=64,name='upconv2',no_bias=False)
    upconv2 = mx.sym.LeakyReLU(data = upconv2,act_type = 'leaky',slope  = 0.1  )

    upsample_pr3to2 = mx.sym.Deconvolution(pr3,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=2,name='upsample_pr3to2',no_bias=False)

    concat_tmp4 = mx.sym.Concat(upsample_pr3to2,upconv2,conv2)

    iconv2 = mx.sym.Convolution(concat_tmp4,pad = (1,1),kernel = (3,3),stride= (1,1),num_filter = 64,name='iconv2')
    iconv2 = mx.sym.LeakyReLU(data = iconv2,act_type = 'leaky',slope  = 0.1 )
    pr2 = mx.sym.Convolution(iconv2,pad = (1,1),kernel=(3,3),stride = (1,1),num_filter = output_dim,name='pr2')
    if is_sparse:
        loss = SparseRegressionLoss(is_l1=is_l1,loss_scale=loss2_scale)
        loss2 = loss(data=pr2,label = downsample2)
    else:
        if is_l1:
            loss2 = mx.sym.MAERegressionOutput(data = pr2,label = downsample2,grad_scale=loss2_scale,name='loss2')
        else:
            loss2 = mx.sym.LinearRegressionOutput(data = pr2,label = downsample2,grad_scale=loss2_scale,name='loss2')

    upconv1 = mx.sym.Deconvolution(iconv2,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter = 32,name='upconv1',no_bias=False)
    upconv1 = mx.sym.LeakyReLU(data = upconv1,act_type = 'leaky',slope  = 0.1 )

    upsample_pr2to1 = mx.sym.Deconvolution(pr2,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=2,name='upsample_pr2to1',no_bias=False)

    concat_tmp5 = mx.sym.Concat(upsample_pr2to1,upconv1,conv1)
    iconv1 = mx.sym.Convolution(concat_tmp5,pad=(1,1),kernel = (3,3),stride=(1,1),num_filter=32,name='iconv1')
    iconv1 = mx.sym.LeakyReLU(data = iconv1,act_type = 'leaky',slope  = 0.1  )

    pr1 = mx.sym.Convolution(iconv1,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name='pr1')
    if is_sparse:
        loss = SparseRegressionLoss(is_l1=is_l1,loss_scale=loss1_scale)
        loss1 = loss(data=pr1,label = downsample1)
    else:
        if is_l1:
            loss1 = mx.sym.MAERegressionOutput(data = pr1,label = downsample1,grad_scale=loss1_scale,name='loss1')
        else:
            loss1 = mx.sym.LinearRegressionOutput(data = pr1,label = downsample1,grad_scale=loss1_scale,name='loss1')

    # upconv0 = mx.sym.Deconvolution(iconv1,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter = 32,name='upconv0',no_bias=False)
    # upconv0 = mx.sym.LeakyReLU(data=upconv0, act_type='leaky', slope=0.1)
    #
    # upsample_pr1to0 = mx.sym.Deconvolution(pr1, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=2,name='upsample_pr1to0',no_bias=False)
    #
    # concat_tmp6 = mx.sym.Concat(upsample_pr1to0, upconv0, concat1)
    # iconv0 = mx.sym.Convolution(concat_tmp6, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=32, name='iconv0')
    # iconv0 = mx.sym.LeakyReLU(data=iconv0, act_type='leaky', slope=0.1)
    #
    # pr0 = mx.sym.Convolution(iconv0, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=output_dim, name='pr0')
    #
    # if is_sparse:
    #     loss = SparseRegressionLoss(is_l1=is_l1, loss_scale=loss1_scale)
    #     loss0 = loss(data=pr0, label=downsample0)
    # else:
    #     if is_l1:
    #         loss0 = mx.sym.MAERegressionOutput(data=pr0, label=downsample0, grad_scale=loss0_scale, name='loss0')
    #     else:
    #         loss0 = mx.sym.LinearRegressionOutput(data=pr0, label=downsample0, grad_scale=loss0_scale, name='loss0')

    # dispnet and flownet have 6 L1 loss layers
    net = mx.sym.Group([loss1,loss2,loss3,loss4,loss5,loss6])
    return net