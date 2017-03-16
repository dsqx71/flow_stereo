import mxnet as mx
import util

class EndPointErr(mx.metric.EvalMetric):
    """
        euclidean distance
    """
    def __init__(self):

        super(EndPointErr, self).__init__('End Point Error')

    def update(self, gt_ndy, pred_ndy):

        gt = gt_ndy[0].as_in_context(pred_ndy[0].context)
        pred = pred_ndy[0]

        # valid : True , NaN : False
        mask = (gt == gt).asnumpy().astype('bool')
        mask = mask[:, 0, :, :]

        # Euclidean distance
        r = (pred - gt) ** 2
        r = mx.nd.sum_axis(r, 1) ** 0.5
        r = r.asnumpy()

        error_mean = r[mask].sum() / mask.sum()
        self.sum_metric += error_mean
        self.num_inst += 1

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
