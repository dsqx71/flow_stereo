import mxnet as mx
import numpy as np

class SparseRegressionLoss(mx.operator.CustomOp):
    """
        SparseRegressionLoss ignore labels with values of NaN
    """
    def __init__(self, loss_scale, is_l1):
        # watch out mxnet serialization problem
        loss_scale = float(loss_scale)
        is_l1 = bool(is_l1)
        self.loss_scale = loss_scale
        self.is_l1 = is_l1

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        y = out_data[0]
        self.assign(y, req[0], x)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        label = in_data[1]
        y = out_data[0]
        # find valid labels
        mask = (label == label)
        # total number of valid points:
        normalize_coeff = mx.nd.sum(mask).asnumpy()[0] / y.shape[1]
        if self.is_l1:
            # L1 loss
            # mx.nd.sign will return 0, if input is nan
            tmp = mx.nd.sign(y - label) * self.loss_scale / float(normalize_coeff)
        else:
            # L2 loss
            tmp = (y - label) * self.loss_scale / float(normalize_coeff)
            # ignore nan
            zeros = mx.nd.zeros(y.shape)
            tmp = mx.nd.where(mask, tmp, zeros)

        self.assign(in_grad[0], req[0], tmp)

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
    # values in disparity map should be positive
    if type == 'stereo':
        data = mx.sym.Activation(data=data, act_type='relu',name=name+'relu')
    # loss
    if  is_sparse:
        loss =mx.symbol.Custom(data=data, label=label, name=name, loss_scale= loss_scale, is_l1=True,
            op_type='SparseRegressionLoss')
    else:
        loss = mx.sym.MAERegressionOutput(data=data, label=label, name=name, grad_scale=loss_scale)

    return (loss,data) if get_input else loss