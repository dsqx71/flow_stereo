import mxnet as mx
import numpy as np
from utils import util

class EndPointErr(mx.metric.EvalMetric):
    """
        euclidean distance
    """
    def __init__(self, which_scale):

        super(EndPointErr, self).__init__('End Point Error_scale:{}'.format(which_scale))
        self.which_scale = which_scale

    def update(self, gt, pred):

        gt = gt[self.which_scale].asnumpy()
        mask = (gt==gt)
        mask = mask[:,0,:,:]
        pred = pred[self.which_scale].asnumpy()
        r = np.power(pred-gt,2)
        r = np.power(r.sum(axis=1),0.5)

        self.sum_metric += r[mask].mean()
        self.num_inst += 1

class D1all(mx.metric.EvalMetric):
    """
       residual > 3  and   residual / gt > 0.05   (defined by kitti)
    """

    def __init__(self):
        super(D1all, self).__init__('D1all')

    def update(self, gt, pred, tau = 3):

        pred = pred[0]
        gt = gt[0]

        pred_all = pred.asnumpy()
        gt_all = gt.asnumpy()

        for i in xrange(gt_all.shape[0]):
            self.sum_metric += util.outlier_sum(pred_all[i][0], gt_all[i][0], tau)
        self.num_inst += gt_all.shape[0]
