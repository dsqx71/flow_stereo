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

def get_conv(name, data, num_filter, kernel, stride, pad, dilate=(1, 1), no_bias=False, with_relu=True, weight=None, is_conv=True,bn=False):

    if is_conv:
        if weight is None:
            conv = mx.symbol.Convolution(name=name, data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                                         pad=pad, dilate=dilate, no_bias=no_bias, workspace=1024)
        else:
            conv = mx.symbol.Convolution(name=name, data=data, num_filter=num_filter, kernel=kernel, stride=stride,
                                         pad=pad, dilate=dilate, no_bias=no_bias, weight=weight, workspace=1024)
    else:
        conv = mx.sym.Deconvolution(name=name, data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                                    no_bias=no_bias)
    if bn:
        conv = mx.symbol.BatchNorm( name=name + '_bn', data=conv, fix_gamma=False, momentum=0.95, eps=2e-5)
    return mx.sym.LeakyReLU(name=name + '_prelu', data=conv, act_type='prelu') if with_relu else conv


def get_feature(data,name,weights):

    # dispnet: 9408 , attention: 11616

    num_filter = 32
    tmp = data = get_conv('level0.0' + name, data, num_filter,
                    kernel=(5, 5), stride=(2, 2), pad=(2, 2), dilate=(1, 1), no_bias=True, weight=weights[0], bn=True)
    data = get_conv('level0.1' + name, data, num_filter,
                    kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), no_bias=True, weight=weights[1], bn=True)
    data0 = data = data + tmp

    # dispnet: 204800 , attention: 202752

    num_filter = 64
    tmp = data = get_conv('level1.0' + name, data, num_filter,
                    kernel=(3, 3), stride=(2, 2), pad=(1, 1), dilate=(1, 1), no_bias=True, weight=weights[2],bn=True)
    tmp2= data = get_conv('level1.1'+name, data, num_filter,
                    kernel=(3, 3), stride=(1, 1),pad=(1, 1), dilate=(1, 1), no_bias=True,weight = weights[3],bn=True)
    data = mx.sym.Concat(tmp,data)
    data = get_conv('level1.2' + name, data, num_filter,
                    kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), no_bias=True, weight=weights[4], bn=True)
    data = mx.sym.Concat(data, tmp2)
    data = get_conv('level1.3' + name, data, num_filter,
                    kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), no_bias=True, weight=weights[5], bn=True)
    data1 = data

    return data0, data1



def block(data,num_filter,name,is_downsample=False,return_regress=False):

    # param = num_filter ^2 * 6 * 3 * 3
    stride = (2,2) if is_downsample else (1,1)
    tmp  = data = get_conv(name+'block.0', data, num_filter, kernel=(3, 3), stride=stride, pad=(1, 1), dilate=(1, 1), bn=True)
    regressor = get_conv(name + 'regressor.0', data, num_filter=1, kernel=(3, 3),
                         stride=(1, 1), pad=(1, 1), dilate=(1, 1), no_bias=False, is_conv=True, with_relu=False)
    data = mx.sym.Concat(data,regressor)

    tmp2 = data = get_conv(name+'block.1', data, num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), bn=True)

    data = mx.sym.Concat(tmp, data)
    data = get_conv(name+'block.2', data, num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), bn=True)

    data = mx.sym.Concat(data, tmp2)
    data = get_conv(name+'block.3', data, num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), dilate=(1, 1), bn=True)

    regressor = get_conv(name + 'regressor.1', data, num_filter=1, kernel=(3, 3),
                         stride=(1, 1), pad=(1, 1), dilate=(1, 1), no_bias=False, is_conv=True, with_relu=False)
    data = mx.sym.Concat(data, regressor)
    if return_regress:
        return data,regressor
    else:
        return data



def corr_net(data0_img1,data0_img2, data1_img1, data1_img2):

    corr0 = mx.sym.CaffeOp(data_0=data0_img1, data_1=data0_img2, num_data=2, name='correlation1/2',
                           prototxt='''
                                    layer{type:\"Correlation1D\"
                                    correlation_param {
                                    pad: 32 kernel_size: 1 max_displacement: 32 stride_1: 1 stride_2: 1
                                    } }''')
    corr0 = mx.sym.LeakyReLU(name='corr0_prelu', data=corr0, act_type='prelu')

    corr1 = mx.sym.CaffeOp(data_0=data1_img1, data_1=data1_img2, num_data=2, name='correlation1/4',
                           prototxt='''
                                    layer{type:\"Correlation1D\"
                                    correlation_param {
                                    pad: 40 kernel_size: 1 max_displacement: 40 stride_1: 1 stride_2: 1
                                    } }''')
    corr1 = mx.sym.LeakyReLU(name='corr1_prelu', data=corr1, act_type='prelu')

    return corr0,corr1

def attention_net(data1,data2,name,num_filter):

    data = mx.sym.Concat(data1,data2)
    data = get_conv(name + 'attention.0', data, num_filter, kernel=(5, 5), stride=(1, 1), pad=(2, 2), dilate=(1, 1),
                    no_bias=False, bn=True)
    score= get_conv(name + 'attention.1', data, 2 , kernel=(5, 5), stride=(1, 1), pad=(2, 2), dilate=(1, 1),
                    no_bias=False, bn=True, with_relu=False)

    score = mx.sym.SoftmaxActivation(data=score, mode='channel',name = name+'softmax')
    score = mx.sym.SliceChannel(score,num_outputs = 2,axis=1)

    data = mx.sym.broadcast_mul(data1,score[0]) + mx.sym.broadcast_mul(data2,score[1])

    return data

def upscale(data,num_filter,name):

    data = get_conv(name + 'deconv', data, num_filter=num_filter, pad=(1, 1), kernel=(4, 4), stride=(2, 2), is_conv=False, with_relu=False)

    return data

def get_loss(data,label,name,is_sparse,is_l1,grad_scale):

    if is_sparse:
        loss = SparseRegressionLoss(is_l1=is_l1, loss_scale=1.0)
        data = loss(data=data, label=label)
    else:
        data = mx.sym.MAERegressionOutput(data=data, label=label, name=name, grad_scale=grad_scale)
    return data

def stereo_net(is_sparse,is_l1):

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')
    downsample0 = mx.sym.Variable('stereo_downsample1')
    downsample1 = mx.sym.Variable('stereo_downsample2')
    downsample2 = mx.sym.Variable('stereo_downsample3')
    downsample3 = mx.sym.Variable('stereo_downsample4')

    weights = [mx.sym.Variable('share{}_weight'.format(i)) for i in range(100)]

    data0_img1, data1_img1 = get_feature(img1, 'img1', weights)
    data0_img2, data1_img2 = get_feature(img2, 'img2', weights)

    corr0_redim = get_conv('corr0_redim', data0_img1, 32, kernel=(1, 1), stride=(1, 1), pad=(0, 0), dilate=(1, 1), bn=True, with_relu=True)
    corr1_redim = get_conv('corr1_redim', data1_img1, 64, kernel=(1, 1), stride=(1, 1), pad=(0, 0), dilate=(1, 1), bn=True, with_relu=True)

    corr0, corr1   = corr_net(data0_img1,data0_img2,data1_img1, data1_img2)

    corr0 = mx.sym.Concat(corr0,corr0_redim)
    corr1 = mx.sym.Concat(corr1,corr1_redim)

    tmp3 = data = block(data=corr0, num_filter=32, name = 'level0', is_downsample=False)
    data, pr3 = block(data=data,  num_filter=64, name = 'level1', is_downsample=True,return_regress=True)
    tmp2 = data
    data = mx.sym.Concat(data,corr1)

    tmp1 = data = block(data=data,num_filter=128, name = 'level3', is_downsample=True)
    data = block(data=data,num_filter=256, name = 'level4', is_downsample=True)
    data = block(data=data, num_filter=256, name='level4b', is_downsample=False)
    data, pr2 = block(data=data, num_filter=256,name = 'level5', is_downsample=False,return_regress=True)

    data = upscale(data=data,num_filter=129, name = 'level6')
    data = attention_net(data1=data, data2=tmp1,num_filter=64,name='level6')
    data = block(data=data, num_filter=128, name='level6', is_downsample=False)

    data = upscale(data=data, num_filter=65, name='level7')
    data = attention_net(data1=data, data2=tmp2, num_filter=32, name='level7')
    data, pr1 = block(data=data, num_filter=64, name='level7', is_downsample=False,return_regress=True)

    data = upscale(data=data, num_filter=33, name='level8')
    data = attention_net(data1=data, data2=tmp3, num_filter=16, name='level8')
    data = block(data=data, num_filter=32, name='level8', is_downsample=False)

    data = upscale(data=data, num_filter=13, name='level9')
    data = mx.sym.Concat(data,img1)
    data = block(data=data, num_filter=8, name='level9', is_downsample=False)
    pr0  = mx.sym.Convolution(data,kernel=(3,3),num_filter = 1,stride=(1,1),pad=(1,1),name='pr')

    loss1 = get_loss(pr0, downsample0,'loss0', is_sparse, is_l1, 0.01)
    loss2 = get_loss(pr1, downsample1,'loss1', is_sparse, is_l1, 0.05)
    loss3 = get_loss(pr2, downsample2,'loss2', is_sparse, is_l1, 0.10)
    loss4 = get_loss(pr3, downsample3,'loss3', is_sparse, is_l1, 0.40)

    net = mx.sym.Group([loss1, loss2, loss3,loss4])
    return net