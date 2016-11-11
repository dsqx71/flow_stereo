import mxnet as mx
import numpy as np

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
        y = out_data[0]
        mask = label > -1e-6
        # mask = np.where(label>1e-6 , (np.abs(y-label) > 3) & (np.abs(y-label)/np.abs(label)>0.05),0)
        dx = in_grad[0]
        if self.is_L1:
            dx[:] = np.sign(y-label)*mask*self.loss_scale
        else:
            dx[:] = (y-label)*mask*self.loss_scale


def get_conv(name, data, num_filter, kernel, stride, pad, dilate=(1, 1), no_bias=False, with_relu=True, weight=None,
             is_conv=True,bn=False):

    if is_conv:
        if weight is None:
            conv = mx.symbol.Convolution(name=name, data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                                         pad=pad, dilate=dilate, no_bias=no_bias,workspace=4096)
        else:
            conv = mx.symbol.Convolution(name=name, data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                                         dilate=dilate, no_bias=no_bias, weight=weight, workspace=4096)
    else:
        conv = mx.sym.Deconvolution(name=name, data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                                    no_bias=no_bias)
    if bn:
        conv = mx.symbol.BatchNorm( name=name + '_bn', data=conv, fix_gamma=False, momentum=0.95, eps=2e-5)
    return mx.sym.LeakyReLU(name=name + '_prelu', data=conv, act_type='prelu') if with_relu else conv


def get_feature(data,name,weights):
    
    num_filter = 64
    index = 0
    data = get_conv('level1.0' + name, data, num_filter,
                    kernel=(5, 5), stride=(2, 2), pad=(2, 2), dilate=(1, 1), no_bias=True, weight=weights[index])
    index += 1
    for  i in range(1, 2):
        data = get_conv('level1.{}'.format(i)+name, data, num_filter,
                    kernel=(3, 3), stride=(1, 1),pad=(1, 1), dilate=(1, 1), no_bias=True,weight = weights[index],bn=True)
        index += 1
    data1 = data

    num_filter = 128
    data = get_conv('level2.0' + name, data, num_filter,
                    kernel=(3, 3), stride=(2, 2), pad=(1, 1), dilate=(1, 1), no_bias=True,weight=weights[index])
    index += 1
    for j in range(1, 4):
        data = get_conv('level2.{}'.format(j)+name, data, num_filter,
                    kernel=(3, 3), stride=(1, 1),pad=(1, 1), dilate=(1, 1), no_bias=True,weight = weights[index],bn=True)
        index += 1
    data2 = data

    num_filter = 256
    data = get_conv('level3.0' + name, data, num_filter,
                    kernel=(3, 3), stride=(2, 2), pad=(1, 1), dilate=(1, 1), no_bias=True, weight=weights[index])
    index += 1
    for j in range(1, 4):
        data = get_conv('level3.{}'.format(j) + name, data, num_filter,
                        kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), no_bias=True,
                        weight=weights[index], bn=True)
        index += 1
    data3 = data

    return data1, data2, data3


def regularizer(data, num_filter, num_layer, name):
    
    for i in range(num_layer):
        data = get_conv(name+'.{}'.format(i), data, num_filter, kernel=(3, 3),
                        stride=(1, 1), pad=(1, 1), dilate=(1, 1), no_bias=False, bn=True)
    regressor = get_conv(name + 'regressor', data, num_filter=1, kernel=(3, 3),
                    stride=(1,1), pad=(1, 1), dilate=(1, 1), no_bias=False, is_conv=True, with_relu=False)
    return data,regressor

def stereo_net(is_sparse,is_l1):

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')
    downsample0 = mx.sym.Variable('stereo_downsample1')
    
    weights = [mx.sym.Variable('share{}_weight'.format(i)) for i in range(20)]
   
    data1_img1, data2_img1, data3_img1 = get_feature(img1, 'img1', weights)
    data1_img2, data2_img2, data3_img2 = get_feature(img2, 'img2', weights)

    corr = mx.sym.CaffeOp(data_0 = data3_img1, data_1= data3_img2, num_data=2, name='correlation',
                          prototxt='''
                                layer{type:\"Correlation1D\"
                                correlation_param {
                                pad: 40 kernel_size: 1 max_displacement: 40 stride_1: 1 stride_2: 1
                                single_direction: 1
                                } }''')
    regressor0 = get_conv('regressor_corr', corr, num_filter = 1, kernel=(3, 3), stride=(1,1), pad=(1, 1), dilate=(1, 1),
                   no_bias=False, is_conv=True, with_relu=False)
    corr = mx.sym.LeakyReLU(name='corr_prelu', data=corr, act_type='prelu')

    data = mx.sym.Concat(corr, data3_img1, regressor0)
    data,regressor1 = regularizer(data, num_filter=512, num_layer=2, name='regularizer0')
    data,regressor2 = regularizer(data, num_filter=256,  num_layer=2, name='regularizer1')
    tmp = mx.sym.Concat(data,regressor0,regressor1,regressor2)
    data = get_conv('deconv1_data', tmp, num_filter=256, pad=(1, 1), kernel=(4, 4), stride = (2,2), is_conv=False,
                    bn=False, with_relu=True)
    regressor = get_conv('deconv1_regressor', tmp, num_filter=1, pad=(1, 1), kernel=(4, 4), stride = (2,2), is_conv=False,
                    bn=False, with_relu=False)

    data = mx.sym.Concat(data2_img1, data , regressor)
    data,regressor0 = regularizer(data, num_filter=128, num_layer=1, name='regularizer2')
    tmp = mx.sym.Concat(data,regressor,regressor0)
    data = get_conv('deconv2_data', tmp, num_filter=64, pad=(1, 1), kernel=(4, 4), stride=(2, 2), is_conv=False,
                    bn=False, with_relu=True)
    regressor = get_conv('deconv2_regressor', tmp, num_filter=1, pad=(1, 1), kernel=(4, 4), stride=(2, 2),
                         is_conv=False, bn=False, with_relu=False)

    data = mx.sym.Concat(data1_img1, data,regressor)
    data,regressor0 = regularizer(data, num_filter=64, num_layer=1, name='regularizer3')
    data = mx.sym.Concat(data,regressor0,regressor)
    data = get_conv('pr', data, num_filter=1, kernel=(3, 3),
                    stride=(1,1), pad=(1, 1), dilate=(1, 1), no_bias=False, is_conv=True, with_relu=False)

    if is_sparse:
        loss = SparseRegressionLoss(is_l1=is_l1, loss_scale=1.0)
        data = loss( data = data, label = downsample0)
    else:
        data = mx.sym.MAERegressionOutput(data=data, label=downsample0, name='loss_regularizer', grad_scale= 1.0)
    return data