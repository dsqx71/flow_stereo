import mxnet as mx
import numpy as np
import cv2
from res_unit import residual_unit

def get_conv(name, data, num_filter, kernel, stride, pad, with_relu,  dilate=(1, 1)):

    conv = mx.symbol.Convolution(name=name, data=data, num_filter=num_filter, kernel=kernel,
                                 stride=stride, pad=pad, dilate=dilate, no_bias=True)
    return (mx.sym.Activation(data=conv, act_type='relu', name=name + '_relu') if with_relu else conv)

# class Warp(mx.operator.CustomOp):
#
#     def forward(self, is_train, req, in_data, out_data, aux):
#
#         img = in_data[0].asnumpy()
#         flow = in_data[1].asnumpy()
#         out = np.zeros_like(img)
#         xv, yv = np.meshgrid(np.arange(img.shape[2]), np.arange(img.shape[3]))
#
#         for i in range(img.shape[0]):
#             mapx = yv.T + flow[i, 0, :, :]
#
#             if flow.shape[1]==2:
#                 mapy = xv.T + flow[i, 1, :, :]
#             else:
#                 mapy = xv.T
#
#             mapx = mapx.astype(np.float32)
#             mapy = mapy.astype(np.float32)
#
#             tmp = cv2.remap(img[i].transpose(1,2,0), mapx, mapy, interpolation=cv2.INTER_LINEAR, borderValue=0)
#             out[i][:] = tmp.transpose(2,0,1)
#
#         assert (out != out).any() == False
#         self.assign(out_data[0], req[0], mx.nd.array(out))
#
#     def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
#         pass
#
# @mx.operator.register("warp")
# class WarpProp(mx.operator.CustomOpProp):
#
#     def __init__(self):
#         super(WarpProp, self).__init__(True)
#
#     def list_arguments(self):
#
#         return ['data', 'flow']
#
#     def list_outputs(self):
#
#         return ['output']
#
#     def infer_shape(self, in_shape):
#
#         img_shape = in_shape[0]
#         flow_shape = in_shape[1]
#         output_shape = in_shape[0]
#
#         assert img_shape[2] ==flow_shape[2] and img_shape[3] ==flow_shape[3]
#
#         return [img_shape, flow_shape], [output_shape], []
#
#     def create_operator(self, ctx, shapes, dtypes):
#
#         return Warp()
#
# class ResampleOp(mx.operator.CustomOp):
#
#     def forward(self, is_train, req, in_data, out_data, aux):
#
#         img = in_data[0].asnumpy()
#         out = []
#         for i in range(img.shape[0]):
#             tmp = cv2.resize(img[i].transpose(1,2,0), (0, 0), fx=2, fy=2)
#             if len(tmp.shape)==2:
#                 out.append(np.expand_dims(tmp,0))
#             else:
#                 out.append(tmp.transpose(2,0,1))
#         out = np.array(out)
#         self.assign(out_data[0], req[0], mx.nd.array(out))
#
#     def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
#         pass
#
# @mx.operator.register("resample")
# class ResampleProp(mx.operator.CustomOpProp):
#
#     def __init__(self):
#
#         super(ResampleProp, self).__init__(True)
#
#     def list_arguments(self):
#
#         return ['data']
#
#     def list_outputs(self):
#         return ['output']
#
#     def infer_shape(self, in_shape):
#
#         img_shape = in_shape[0]
#         output_shape = in_shape[0][:2] + [img_shape[2]*2, img_shape[3]*2]
#         return [img_shape], [output_shape], []
#
#     def create_operator(self, ctx, shapes, dtypes):
#
#         return ResampleOp()

def Gfunction(img1, img2, img3, name, flow = None, factor=1, out_dim=2):

    if out_dim ==1:
        corr1 = mx.sym.Correlation1D(data1=img1, data2=img2, pad_size=1*factor, kernel_size=1, max_displacement=1*factor, stride1=1, stride2=1)
    else:
        corr1 = mx.sym.Correlation(data1=img1, data2=img2, pad_size=1*factor, kernel_size=1, max_displacement=1*factor, stride1=1, stride2=1)

    if out_dim == 1:
        corr2 = mx.sym.Correlation1D(data1=img1, data2=img3, pad_size=1*factor, kernel_size=1, max_displacement=1*factor,
                                     stride1=1, stride2=1)
    else:
        corr2 = mx.sym.Correlation(data1=img1, data2=img3, pad_size=1*factor, kernel_size=1, max_displacement=1*factor, stride1=1,
                                   stride2=1)

    data = mx.sym.Concat(img1, img2, img3, flow) if flow is not None else mx.sym.Concat(img1, img2)
    data = get_conv(name + '.0', data, num_filter=int(32 * factor), kernel=(7, 7), stride=(1, 1), pad=(3, 3), with_relu=True,
                    dilate=(1, 1))
    tmp1 = data = get_conv(name + '.1', data, num_filter=int(32 * factor), kernel=(5, 5), stride=(1, 1), pad=(2, 2), with_relu=True,
                    dilate=(1, 1))
    data = mx.sym.Concat(data, corr1, corr2)
    data = get_conv(name + '.2', data, num_filter=int(64 * factor), kernel=(5, 5), stride=(1, 1), pad=(2, 2), with_relu=True,
                    dilate=(1, 1))
    data = get_conv(name + '.3', data, num_filter=int(64 * factor), kernel=(3, 3), stride=(1, 1), pad=(1, 1), with_relu=True,
                    dilate=(1, 1))
    data = get_conv(name + '.4', data, num_filter=int(32 * factor), kernel=(3, 3), stride=(1, 1), pad=(1, 1), with_relu=True,
                    dilate=(1, 1))
    data = get_conv(name + '.5', data, num_filter=int(32 * factor), kernel=(3, 3), stride=(1, 1), pad=(1, 1), with_relu=True,
                    dilate=(1, 1))
    # data = data + tmp1
    data = get_conv(name + '.6', data, num_filter=int(16 * factor), kernel=(5, 5), stride=(1, 1), pad=(2, 2), with_relu=True,
                    dilate=(1, 1))
    data = get_conv(name + '.7', data, num_filter=int(16 * factor), kernel=(7, 7), stride=(1, 1), pad=(3, 3),
                    with_relu=True,
                    dilate=(1, 1))
    data = mx.sym.Convolution(data, pad=(3, 3), kernel=(7, 7), stride=(1, 1), num_filter=out_dim, name=name+'.predictor')
    data = mx.sym.Activation(data=data, act_type='relu', name=name+'_predictor_relu')
    return data

def block(img1, img2, flow, name, factor=1, out_dim=2,up_sample=True):

    # flow = mx.sym.BlockGrad(data=flow,name=name+'block_flow')
    if up_sample:
        flow = mx.sym.Deconvolution(flow, pad=(1,1), kernel=(4,4), stride=(2,2), num_filter=out_dim,
                                    name=name+'_resample_flow',no_bias=True)
    img2_warped = mx.sym.transpose(data=img2, axes=(0,2,3,1),name=name+'img_transpose')
    flow_tmp = mx.sym.Concat(flow, flow*0)
    flow_tmp = mx.sym.transpose(data=flow_tmp, axes=(0,2,3,1),name=name+'flow_transpose')
    img2_warped = mx.sym.Warp(data=img2_warped,grid=flow_tmp,only_grid=False,name=name+'warp')
    img2_warped = mx.sym.transpose(data=img2_warped, axes=(0, 3, 1, 2), name=name + 'imgwarp_transpose')
    res = Gfunction(img1, img2_warped, img2, name, flow, factor, out_dim=out_dim)
    data = res + flow

    return data

def init(img1, img2, out_dim=2):

    v0 = Gfunction(img1=img1, img2=img2, img3=img2, name='v0', flow=None, factor=8, out_dim=out_dim)

    return v0

def spynet_symbol(type):

    out_dim = 2 if type=='flow' else 1
    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    flow1 = mx.sym.Variable(type+'_downsample1')
    flow2 = mx.sym.Variable(type+'_downsample2')
    flow3 = mx.sym.Variable(type+'_downsample3')
    flow4 = mx.sym.Variable(type+'_downsample4')
    flow5 = mx.sym.Variable(type+'_downsample5')
    flow6 = mx.sym.Variable(type + '_downsample6')
    flow7 = mx.sym.Variable(type + '_downsample7')

    img1_downsample = []
    img2_downsample = []

    for i in range(6):
        if i == 0:
            img1_tmp = img1
            img2_tmp = img2
        else:
            img1_tmp = img1_downsample[-1]
            img2_tmp = img2_downsample[-1]
        img1_conv = get_conv(name='conv_img1{}'.format(i), data=img1_tmp, num_filter=2**(i+3), kernel=(5,5), stride=(2,2), pad=(2,2),
                             with_relu=True, dilate=(1, 1))
        img1_pool = mx.sym.Pooling(data=img1_tmp, kernel=(2, 2), stride=(2, 2), pool_type='avg')
        img1_downsample.append(mx.sym.Concat(img1_conv, img1_pool))

        img2_conv = get_conv(name='conv_img2{}'.format(i), data=img2_tmp, num_filter=2 ** (i + 3), kernel=(5, 5),
                             stride=(2, 2), pad=(2, 2), with_relu=True, dilate=(1, 1))
        img2_pool = mx.sym.Pooling(data=img2_tmp, kernel=(2, 2), stride=(2, 2), pool_type='avg')
        img2_downsample.append(mx.sym.Concat(img2_conv, img2_pool))


    v0 = init(img1_downsample[5], img2_downsample[5], out_dim=out_dim)
    v1 = block(img1_downsample[4], img2_downsample[4], v0, name='v1', factor=4, out_dim=out_dim)
    v2 = block(img1_downsample[3], img2_downsample[3], v1, name='v2', factor=2, out_dim=out_dim)
    v3 = block(img1_downsample[2], img2_downsample[2], v2, name='v3', factor=1, out_dim=out_dim)
    v4 = block(img1_downsample[1], img2_downsample[1], v3, name='v4', factor=1, out_dim=out_dim)
    v5 = block(img1_downsample[0], img2_downsample[0], v4, name='v5', factor=1, out_dim=out_dim)
    v6 = block(img1, img2, v5, name='v6', factor=1, out_dim=out_dim)

    loss6 = mx.sym.MAERegressionOutput(data=v6, label=flow1 / 1, name='loss6', grad_scale=1.00)
    loss5 = mx.sym.MAERegressionOutput(data=v5, label=flow2 / 2, name='loss5', grad_scale=1.00)
    loss4 = mx.sym.MAERegressionOutput(data=v4, label=flow3 / 4, name='loss4', grad_scale=1.00)
    loss3 = mx.sym.MAERegressionOutput(data=v3, label=flow4 / 8, name='loss3', grad_scale=1.00)
    loss2 = mx.sym.MAERegressionOutput(data=v2, label=flow5 / 16, name='loss2', grad_scale=1.00)
    loss1 = mx.sym.MAERegressionOutput(data=v1, label=flow6 / 32, name='loss1', grad_scale=0.50)
    loss0 = mx.sym.MAERegressionOutput(data=v0, label=flow7 / 64, name='loss0', grad_scale=0.30)

    loss = mx.sym.Group([loss6,loss5,loss4,loss3,loss2,loss1,loss0])
    # loss = mx.sym.Group([loss6,loss5,los])
    return loss







