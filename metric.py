import mxnet as mx
import numpy as np

class EndPointErr(mx.metric.EvalMetric):
    """
        euclidean distance
    """

    def __init__(self):
        super(EndPointErr, self).__init__('End Point Error')

    def update(self, pred, gt):

        r = (pred - gt).asnumpy()
        r = np.power(r, 2)
        r = np.sqrt(r.sum(axis=1))
        self.sum_metric += r.mean()
        self.num_inst += 1.0


class D1all(mx.metric.EvalMetric):
    """
       residual > 3  and   residual / gt > 0.05
    """

    def __init__(self):
        super(D1all, self).__init__('D1all')

    def update(self, pred, gt):

        outlier = np.zeros(gt.shape)
        mask = gt > 0
        gt = np.round(gt[mask] / 256.0)
        pred = pred[mask]
        err = np.abs(pred - gt)
        outlier[mask] = err

        self.sum_metric += (err[err>tau]/gt[err>tau].astype(np.float32) > 0.05).sum()/float(mask.sum())
        self.num_inst += gt.shape[0]

