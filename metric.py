import mxnet as mx
import numpy as np
from utils import util
import matplotlib.pyplot as plt

class EndPointErr(mx.metric.EvalMetric):
    """
        euclidean distance
    """
    def __init__(self, which_scale):

        super(EndPointErr, self).__init__('End Point Error_scale:{}'.format(which_scale))
        self.which_scale = which_scale

    def update(self, gt_ndy, pred_ndy):


        # mask1 = (label!=label)
        # label[mask1] = y[mask1]
        # # ignore sub-pixel error
        # mask2 = (np.abs(y-label) < 0.4)
        # # ignore too large displacement
        # mask3 = label > 250
        # mask4 = label > 450
        # # pay attention to large error
        # mask5 = np.abs(label-y) > 3
        # mask6 = np.abs(label-y) > 10
        # # pay attention to small displacement
        # mask7 = label < 20
        # tmp = np.ones_like(label)
        # tmp[mask1] = 0
        # tmp[mask2] = tmp[mask2] * 0.01
        # tmp[mask3] = tmp[mask3] * 0.5
        # tmp[mask4] = tmp[mask4] * 0.2
        # tmp[mask5] = tmp[mask5] * 3.0
        # tmp[mask6] = tmp[mask6] * 3.0
        # tmp[mask7] = tmp[mask7] * 2.0
        # plt.figure()
        # plt.imshow(tmp[0][0])
        # plt.colorbar()
        # # plt.waitforbuttonpress()
        for i in range(0, 1):
            plt.figure()
            plt.imshow(pred_ndy[i].asnumpy()[0,0])
            plt.colorbar()
            plt.waitforbuttonpress()

        label = gt_ndy[0].asnumpy()
        y = pred_ndy[0].asnumpy()
        mask_outlier = (np.abs(label - y) >= 3) & (np.abs(label - y) / (label + 1E-2) >= 0.05)
        plt.figure()
        plt.imshow(mask_outlier[0,0])
        plt.waitforbuttonpress()

        gt = gt_ndy[self.which_scale].asnumpy()
        mask = (gt==gt)
        mask = mask[:,0,:,:]
        pred = pred_ndy[self.which_scale].asnumpy()
        # r = np.abs(pred-gt)
        r = np.power(pred-gt,2)
        r = np.power(r.sum(axis=1),0.5)
        # plt.figure()
        # plt.imshow(r[0])
        # plt.colorbar()
        # plt.waitforbuttonpress()
        #
        # plt.figure()
        # plt.imshow(r[0]>0.5)
        # plt.colorbar()
        # plt.waitforbuttonpress()

        # print r[0].mean(), r.mean()
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
            tmp, outlier = util.outlier_sum(pred_all[i][0], gt_all[i][0], tau)
            self.sum_metric += tmp
        self.num_inst += gt_all.shape[0]
