import mxnet as mx
import numpy as np
import cv2

def get_conv(name, data, num_filter, kernel, stride, pad, with_relu,  dilate=(1, 1)):

    conv = mx.symbol.Convolution(name=name, data=data, num_filter=num_filter, kernel=kernel,
                                 stride=stride, pad=pad, dilate=dilate, no_bias=True)

    return (mx.symbol.LeakyReLU(name=name + '_prelu', act_type='prelu', data=conv) if with_relu else conv)

class Warp(mx.operator.CustomOp):

    def forward(self, is_train, req, in_data, out_data, aux):

        img = in_data[0].asnumpy()
        flow = in_data[1].asnumpy()
        out = np.zeros_like(img)

        xv, yv = np.meshgrid(np.arange(img.shape[2]), np.arange(img.shape[3]))
        for i in range(img.shape[0]):
            mapx = yv.T + flow[i, 0, :, :]
            mapy = xv.T + flow[i, 1, :, :]

            mapx = mapx.astype(np.float32)
            mapy = mapy.astype(np.float32)

            tmp = cv2.remap(img[i].transpose(1,2,0), mapx, mapy, interpolation=cv2.INTER_LINEAR, borderValue=0)
            out[i][:] = tmp.transpose(2,0,1)

        self.assign(out_data[0], req[0], mx.nd.array(out))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass

@mx.operator.register("warp")
class WarpProp(mx.operator.CustomOpProp):

    def __init__(self):
        super(WarpProp, self).__init__(True)

    def list_arguments(self):

        return ['data', 'flow']

    def list_outputs(self):

        return ['output']

    def infer_shape(self, in_shape):

        img_shape = in_shape[0]
        flow_shape = in_shape[1]
        output_shape = in_shape[0]

        return [img_shape, flow_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):

        return Warp()

class ResampleOp(mx.operator.CustomOp):

    def __init__(self, width, height):

        self.width = width
        self.height = height

    def forward(self, is_train, req, in_data, out_data, aux):

        img = in_data[0].asnumpy()
        out = np.zeros(img.shape[:2] + (self.height, self.width))
        for i in range(img.shape[0]):
            tmp = cv2.resize(img[i].transpose(1,2,0),(self.width, self.height))
            out[i][:] = tmp.transpose(2,0,1)
        self.assign(out_data[0], req[0], mx.nd.array(out))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        pass

@mx.operator.register("resample")
class ResampleProp(mx.operator.CustomOpProp):

    def __init__(self, height, width):

        super(ResampleProp, self).__init__(True)
        self.width = int(width)
        self.height = int(height)

    def list_arguments(self):

        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):

        img_shape = in_shape[0]
        output_shape = in_shape[0][:2] + [self.height, self.width]
        return [img_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):

        return ResampleOp(self.width, self.height)

def Gfunction(img1, img2, name, factor=1):

    data = mx.sym.Concat(img1, img2)
    data = get_conv(name+'.0', data, num_filter= 32*factor, kernel = (7, 7), stride= (1, 1), pad=(3, 3), with_relu=True, dilate=(1, 1))
    data = get_conv(name+'.1', data, num_filter= 64*factor, kernel = (7, 7), stride= (1, 1), pad=(3, 3), with_relu=True, dilate=(1, 1))
    data = get_conv(name+'.2', data, num_filter= 32*factor, kernel = (7, 7), stride= (1, 1), pad=(3, 3), with_relu=True, dilate=(1, 1))
    data = get_conv(name+'.3', data, num_filter= 16*factor, kernel = (7, 7), stride= (1, 1), pad=(3, 3), with_relu=True, dilate=(1, 1))
    data = get_conv(name+'.predictor', data, num_filter= 2, kernel = (7, 7), stride= (1,1), pad=(3,3), with_relu=False, dilate=(1, 1))

    return data

def block(img1, img2, flow, height, width, name, factor=1):

    img1 = mx.symbol.Custom(data=img1, name=name+'_resample_img1',op_type='resample', height=height, width=width)
    img2 = mx.symbol.Custom(data=img2, name=name+'_resample_img2',op_type='resample', height=height, width=width)
    flow = mx.symbol.Custom(data=flow, name=name+'_resample_flow',op_type='resample', height=height, width=width)
    img2_warped = mx.sym.Custom(data=img2, flow = flow, name=name+'_warp',op_type='warp')
    res = Gfunction(img1, img2_warped, name, factor)
    data = res + flow

    return data

def init(img1, img2, height, width):

    img1 = mx.symbol.Custom(data=img1, name='v0_resample_img1', op_type='resample', height=height, width=width)
    img2 = mx.symbol.Custom(data=img2, name='v0_resample_img2', op_type='resample', height=height, width=width)

    v0 = Gfunction(img1, img2, 'v0',4)

    return v0

def spynet_symbol(height, width):

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')
    flow1 = mx.sym.Variable('flow_downsample1')
    flow2 = mx.sym.Variable('flow_downsample2')
    flow3 = mx.sym.Variable('flow_downsample3')
    flow4 = mx.sym.Variable('flow_downsample4')
    flow5 = mx.sym.Variable('flow_downsample5')
    flow6 = mx.sym.Variable('flow_downsample6')

    # img1_tmp = mx.symbol.Custom(data=img1, name='try1',op_type='resample', height=height/2, width=width/2)
    # img2_tmp = mx.symbol.Custom(data=img2, name='try2',op_type='resample', height=height/2, width=width/2)
    # flow = mx.symbol.Custom(data=flow3/4, name='flow1',op_type='resample', height=height/2, width=width/2)
    # flow = flow*2
    # img2_warped = mx.sym.Custom(data=img2_tmp, flow=flow, name='just_warp', op_type='warp')
    #
    # img1_tmp = mx.sym.BlockGrad(data=img1_tmp,name='hehe')
    # img2_warp = mx.sym.BlockGrad(data=img2_warped,name='hehehe')
    # img2_tmp = mx.sym.BlockGrad(data=img2_tmp)
    # flow = mx.sym.BlockGrad(data=flow,name='daffadsf')

    v0 = init(img1, img2, height/32, width/32)

    v1 = block(img1, img2, v0*2, height/16, width/16, name = 'v1', factor=2)
    v2 = block(img1, img2, v1*2, height/8, width/8, name ='v2', factor=1)
    v3 = block(img1, img2, v2*2, height/4, width/4, name = 'v3', factor=1)
    v4 = block(img1, img2, v3*2, height/2, width/2, name ='v4', factor=1)
    v5 = block(img1, img2, v4*2, height, width, name ='v5', factor=1)

    loss1 = mx.sym.MAERegressionOutput(data=v5, label=flow1, name='loss5', grad_scale=0.0)
    loss2 = mx.sym.MAERegressionOutput(data=v4, label=flow2/2, name='loss4', grad_scale=0.0)
    loss3 = mx.sym.MAERegressionOutput(data=v3, label=flow3/4, name='loss3', grad_scale=0.0)
    loss4 = mx.sym.MAERegressionOutput(data=v2, label=flow4/8, name='loss2', grad_scale=0.0)
    loss5 = mx.sym.MAERegressionOutput(data=v1, label=flow5/16, name='loss1', grad_scale=0.0)
    loss6 = mx.sym.MAERegressionOutput(data=v0, label=flow6/32, name='loss0', grad_scale=1.00)

    loss = mx.sym.Group([loss1, loss2, loss3, loss4, loss5, loss6])
    return loss







