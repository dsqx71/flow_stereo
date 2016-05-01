import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from numba import  autojit

class EndPointErr(mx.metric.EvalMetric):
    """
        euclidean distance:    sqrt((u_pred-u_label)^2 + (v_pred-v_label)^2)
    """
    def __init__(self):
        super(EndPointErr, self).__init__('End Point Error')

    @autojit
    def update(self, pred, gt,plot_err=False):

        r = (pred - gt).asnumpy()
        r = np.power(r, 2)
        r = np.sqrt(r.sum(axis=1))
        if plot_err :
            plt.figure()
            plt.imshow(r[0])
            plt.colorbar()
            plt.title('Err')
        self.sum_metric += r.mean()
        self.num_inst += 1.0


class D1all(mx.metric.EvalMetric):
    """
       residual > 3  and   residual / gt > 0.05   (defined by kitti)
    """

    def __init__(self):
        super(D1all, self).__init__('D1all')

    @autojit
    def update(self, pred, gt,plot=False,tau = 3):
        pred = pred.asnumpy()[0][0]
        gt   = gt.asnumpy()[0][0]
        outlier = np.zeros(gt.shape)
        mask = gt > 0
        gt = np.round(gt[mask] / 256.0)
        pred = pred[mask]
        err = np.abs(pred - gt)
        outlier[mask] = err

        if plot:
            plt.figure()
            plt.imshow(outlier)
            plt.title('Outlier')
        self.sum_metric += (err[err>tau]/gt[err>tau].astype(np.float32) > 0.05).sum()/float(mask.sum())
        self.num_inst += gt.shape[0]

