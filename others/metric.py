import mxnet as mx
import util
import visualize
import numpy as np
import cv2

class EndPointErr(mx.metric.EvalMetric):
    """
        euclidean distance:    sqrt((u_pred-u_label)^2 + (v_pred-v_label)^2)
    """
    def __init__(self):
        super(EndPointErr, self).__init__('End Point Error')

    def update(self, gt, pred):

        # visualize.plot(pred[5].asnumpy()[0, 0], 'img')
        # visualize.plot(pred[0].asnumpy()[0, 0], 'prediction')
        # visualize.plot(pred[2].asnumpy()[0, 0], 'map')
        # visualize.plot(pred[3].asnumpy()[0, 0], 'replace')
        # visualize.plot(pred[4].asnumpy()[0, 0], 'refine')

        pred = pred[0].asnumpy()
        gt = gt[0].asnumpy()

        mask = (gt == gt)[:, 0, :, :]
        r = pred - gt
        r = cv2.pow(r, 2)
        r = cv2.pow(r.sum(axis=1), 0.5)

        self.sum_metric += r[mask].sum()
        self.num_inst += mask.sum()

class D1all(mx.metric.EvalMetric):
    """
       outlier : abs(y-label) > 3  &&  abs(y-label) / label > 0.05
    """

    def __init__(self):
        super(D1all, self).__init__('D1all')

    def update(self, gt, pred, tau = 3):

        pred = pred[0]
        gt = gt[0]

        pred_all = pred.asnumpy()
        gt_all = gt.asnumpy()

        for i in range(gt_all.shape[0]):
            tmp = util.outlier_sum(pred_all[i], gt_all[i], tau)
            self.sum_metric += tmp
        self.num_inst += gt_all.shape[0]
