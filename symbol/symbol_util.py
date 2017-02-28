import mxnet as mx
import numpy as np

class SparseRegressionLoss(mx.operator.CustomOp):
    """
        SparseRegressionLoss will only calculate gradients where the values of label are not NaN
    """
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

def get_loss(data, label, loss_scale, name, get_input=False, is_sparse = False, type='stereo'):

    # disparity should be positive
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

    return (loss,data) if get_input else loss