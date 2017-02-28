from symbol.enet_symbol import  get_conv

from symbol.sym_delete.res_unit import residual_unit


def detect(data, output_dim, name):

    data = get_conv(name='detect.0'+name, data=data, num_filter=32, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9)
    data = mx.sym.Pooling(data=data,kernel=(2, 2),pool_type='max',stride=(2,2),name='detect.pool.0'+name)
    tmp2 = data = get_conv(name='detect.1'+name, data=data, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9)
    data = mx.sym.Pooling(data=data, kernel=(2, 2), pool_type='max', stride=(2, 2), name='detect.pool.1'+name)
    data = get_conv(name='detect.2'+name, data=data, num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9)
    data = get_conv(name='detect.3'+name, data=data, num_filter=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9)
    data = get_conv(name='detect.4'+name, data=data, num_filter=64, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + tmp2
    data = get_conv(name='detect.5'+name, data=data, num_filter=32, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = get_conv(name='detect.6'+name, data=data, num_filter=output_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=False, bn_momentum=0.9, is_conv=True)
    data = mx.sym.Activation(data=data,act_type='sigmoid',name='predict_error_map'+name)

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
    data = get_conv(name='replace.6'+name, data=data, num_filter=512, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder4
    data = get_conv(name='replace.7'+name, data=data, num_filter=512, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder3
    data = get_conv(name='replace.8'+name, data=data, num_filter=256, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder2
    data = get_conv(name='replace.9'+name, data=data, num_filter=128, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder1
    data = get_conv(name='replace.10'+name, data=data, num_filter=64, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder0
    data = get_conv(name='replace.11'+name, data=data, num_filter=output_dim, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
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

    data = get_conv(name='refine.decoder0' + name, data=data, num_filter=256, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
             with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder2
    data = get_conv(name='refine.decoder1' + name, data=data, num_filter=128, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder1
    data = get_conv(name='refine.decoder2' + name, data=data, num_filter=64, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder0
    data = get_conv(name='refine.decoder3' + name, data=data, num_filter=output_dim, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
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




import mxnet as mx
import numpy as np

class SparseRegressionLoss(mx.operator.CustomOp):

    def __init__(self,loss_scale, is_l1):

        self.loss_scale = loss_scale
        self.is_l1 = is_l1

    def forward(self, is_train, req, in_data, out_data, aux):

        x = in_data[0]
        y = out_data[0]
        self.assign(y, req[0], x)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        label = in_data[1].asnumpy()
        y = out_data[0].asnumpy()
        # find NaN
        mask_nan = (label != label)
        normalize_coeff = (~mask_nan).sum()
        if self.is_l1:
            tmp = np.sign(y - label) * self.loss_scale / float(normalize_coeff)
        else:
            tmp = (y - label) * self.loss_scale / float(normalize_coeff)
        # ignore NaN
        tmp[mask_nan] = 0
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

def conv_unit(sym, name, weights, bias):

    conv1 = mx.sym.Convolution(data=sym,pad=(3, 3), kernel=(7, 7),stride=(2, 2),num_filter=64,
                               weight=weights[0], bias=bias[0], name='conv1' + name)
    conv1 = mx.sym.LeakyReLU(data = conv1,  act_type = 'leaky', slope  = 0.1 )

    conv2 = mx.sym.Convolution(data = conv1, pad  = (2,2),  kernel=(5,5),stride=(2,2),num_filter=128,
                                 weight = weights[1], bias = bias[1], name='conv2' + name)
    conv2 = mx.sym.LeakyReLU(data = conv2, act_type = 'leaky', slope = 0.1)

    return conv1,conv2,

def stereo_net(net_type='stereo', is_sparse = False,
               loss0_scale=0.00,
               loss1_scale=0.20, loss2_scale=0.10,
               loss3_scale=0.005, loss4_scale=0.0,
               loss5_scale=0.00, loss6_scale=0.0):

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')
    if net_type == 'stereo':
        output_dim = 1
    else:
        output_dim = 2

    # downsample0 = mx.sym.Variable(net_type + '_downsample1')
    downsample1 = mx.sym.Variable(net_type + '_downsample1')
    downsample2 = mx.sym.Variable(net_type + '_downsample2')
    downsample3 = mx.sym.Variable(net_type + '_downsample3')
    downsample4 = mx.sym.Variable(net_type + '_downsample4')
    downsample5 = mx.sym.Variable(net_type + '_downsample5')
    downsample6 = mx.sym.Variable(net_type + '_downsample6')

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
    # pr = mx.sym.Deconvolution(pr1, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=1, name='init_label.drr',
    #                           no_bias=True)
    # data = mx.sym.Concat(img1, img2)
    # pr_final, map, replace_value, refine_value = detect_replace_refine(init_label=pr, input_feature=data, output_dim=output_dim, name='drr')

    # loss = get_loss(pr_final, downsample0, 1.00, name='loss', get_data=False, is_sparse=is_sparse, type=net_type)
    # loss0 = get_loss(-pr, downsample0, 1.00, name='loss0', get_data=False, is_sparse = is_sparse,type=net_type)
    loss1, data = get_loss(-pr1, downsample1, 1.00, name='loss1', get_data=True, is_sparse = is_sparse,type=net_type)
    loss2 = get_loss(pr2, downsample2, 0.10 , name='loss2', get_data=False, is_sparse = is_sparse,type=net_type)
    loss3 = get_loss(pr3, downsample3, 0.0, name='loss3', get_data=False, is_sparse = is_sparse,type=net_type)
    loss4 = get_loss(pr4, downsample4, 0.0, name='loss4', get_data=False, is_sparse = is_sparse,type=net_type)
    loss5 = get_loss(pr5, downsample5, 0.00, name='loss5', get_data=False, is_sparse = is_sparse,type=net_type)
    loss6 = get_loss(pr6, downsample6, 0.00, name='loss6', get_data=False, is_sparse = is_sparse,type=net_type)

    # map = mx.sym.BlockGrad(data=map,name='map_tmp')
    # replace = mx.sym.BlockGrad(data=replace_value,name='replace')
    # refine = mx.sym.BlockGrad(data=refine_value,name='refine')
    # net = mx.sym.Group([loss, map,replace,refine])
    data = mx.sym.BlockGrad(data=data,name='tmp_pr1')
    net = mx.sym.Group([data])
    return data

def get_loss(data,label,loss_scale,name,get_data=False, is_sparse = False, type='stereo'):

    if type == 'stereo':
        data = mx.sym.Activation(data=data, act_type='relu',name=name+'relu')

    if  is_sparse:
        loss =mx.symbol.Custom(data=data, label=label, name=name, loss_scale= loss_scale, is_l1=True,
            op_type='SparseRegressionLoss')
        # loss = mx.symbol.CaffeLoss(data=data, label=label,
        #                            name = 'loss_caffe',
        #                            prototxt = '''
        #                            layer {
        #                               type: "L1Loss"
        #                                loss_weight: %f
        #                                l1_loss_param {
        #                                l2_per_location: false
        #                                normalize_by_num_entries: true
        #                               }
        #                            }''' % loss_scale)
    else:
        loss = mx.sym.MAERegressionOutput(data=data, label=label, name=name, grad_scale=loss_scale)

    return (loss,data) if get_data else loss


