import mxnet as mx
import numpy as np
from utils import util
import matplotlib.pyplot as plt

class EndPointErr(mx.metric.EvalMetric):
    """
        euclidean distance
    """
    def __init__(self):

        super(EndPointErr, self).__init__('End Point Error')

    def update(self, gt_ndy, pred_ndy):

        # plt.figure()
        # plt.imshow(pred_ndy[0].asnumpy()[0,0])
        # plt.colorbar()
        # plt.waitforbuttonpress()
        #
        # plt.figure()
        # plt.imshow(gt_ndy[0].asnumpy()[0, 0])
        # plt.colorbar()
        # plt.waitforbuttonpress()

        # gt = gt_ndy[self.which_scale].asnumpy()
        # mask = (gt==gt)
        # mask = mask[:,0,:,:]
        # pred = pred_ndy[self.which_scale].asnumpy()
        #
        # r = np.power(pred-gt,2)
        # r = np.power(r.sum(axis=1),0.5)

        # self.sum_metric += r[mask].mean()
        self.sum_metric += pred_ndy[1].asnumpy()[0]
        self.num_inst += 1

class D1all(mx.metric.EvalMetric):
    """
       outlier : abs(y-label) > 3  &&  abs(y-label) / label > 0.05
       D1: Percentage of stereo disparity outliers in first frame
    """

    def __init__(self):
        super(D1all, self).__init__('D1all')

    def update(self, gt, pred, tau = 3):

        pred = pred[0]
        gt = gt[0]

        pred_all = pred.asnumpy()
        gt_all = gt.asnumpy()

        for i in range(gt_all.shape[0]):
            tmp, outlier = util.outlier_sum(pred_all[i][0], gt_all[i][0], tau)
            self.sum_metric += tmp
        self.num_inst += gt_all.shape[0]
