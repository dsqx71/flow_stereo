import mxnet as mx
import logging
from collections import namedtuple
DataBatch = namedtuple('DataBatch', ['data', 'label'])

class RNNModule(mx.module.BaseModule):

    def __init__(self, cnn_sym, rnn_sym, data_names=None, label_names=None,
                 logger=logging, context=mx.context.cpu(), train_cnn=False,
                 work_load_list=None, fixed_param_names=None,contain_rnn_params=False):

        super(RNNModule, self).__init__(logger=logger)


    def bind(self,*args,**kwargs):

        self._rnn_module.bind(data_shapes=self._cnn_module.output_shapes+data_shapes[2:], label_shapes=label_shapes,
                              for_training=for_training, inputs_need_grad=self._train_cnn)
