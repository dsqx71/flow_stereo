import atexit
import logging
import multiprocessing as mp
import os
from collections import namedtuple
from math import exp
from random import shuffle

import Queue
import mxnet as mx
import numpy as np

from .config import cfg
from .data_util import resize

DataBatch = namedtuple('DataBatch', ['data', 'label', 'pad', 'index'])
class numpyloader(mx.io.DataIter):
    """
    A dataloader implemented by numpy and opencv2

    Notes:
     - Suppose that the batch size will not change during the entire training process
     - Assume that output_name and label_name have the same prefix, output_name ends with '_output',
      label_name ends with '_label', like 'loss1_output', 'loss1_label'

    Parameters
    ----------
    experiment_name: str,
        name of experiment
    dataset : DataSet instance
    augmentation : Augmentation instance
    batch_shape : tuple of int,
        input shape
    label_shape : list of tuples which are in the form of (output name, output shape)
    n_thread : int,
        number of threads used for reading and preprocessing data
    be_shuffle : bool
        whether data should be shuffled
    interpolation_method : str,
        allowable values are 'neareast' and 'bilinear'
    use_rnn : bool
        whether to provide hidden states of rnn
    rnn_hidden_shapes: None or list of tuple with the form of (hidden state name, shape of hidden state),
    initial_coeff : float,
        initial discount of augmentation coefficients
        default is 0.5, the allowable values should be within [0, 1]
    final_coeff : float,
        final discount of augmentation coeffcients
        default is 1.0, the allowable values should be within [0, 1]
    half_life : float
        indicates how fast the discount of augmentation coefficients changes, as the number of iterations increases.
    """
    def __init__(self,
                 experiment_name,
                 dataset,
                 augmentation,
                 batch_shape,
                 label_shape,
                 n_thread=40,
                 be_shuffle=True,
                 interpolation_method='nearest',
                 use_rnn=False,
                 rnn_hidden_shapes=None,
                 initial_coeff=0.1,
                 final_coeff=1.0,
                 half_life=50000):

        super(numpyloader, self).__init__()
        # shapes
        self.batch_shape = batch_shape # (batchsize, channel, height, width)
        self.target_shape = batch_shape[2:] # target_height, target_width of input
        self.label_shape = label_shape # dict of label names and label shapes

        # preprocessing
        self.interpolation_method = interpolation_method
        self.augmentation = augmentation
        assert 'bilinear' == interpolation_method or 'nearest' == interpolation_method, 'wrong interpolation method'

        # setting of data
        self.be_shuffle = be_shuffle
        self.data_dirs = dataset.dirs
        self.data_num = len(self.data_dirs)
        self.get_data = dataset.get_data
        self.data_type = dataset.data_type
        self.current_index = 0
        self.pad = self.data_num % self.batch_shape[0]
        self.experiment_name = experiment_name
        assert self.data_type == 'stereo' or self.data_type == 'flow', 'wrong data type'

        # setting of multi-process
        self.stop_word = '==STOP--'
        self.n_thread = n_thread
        self.worker_proc = None
        self.stop_flag = mp.Value('b', False)
        self.result_queue = mp.Queue(maxsize=self.batch_shape[0]*20)
        self.data_queue = mp.Queue()

        # load mean and num of iterations
        if os.path.isfile(cfg.dataset.mean_dir + self.experiment_name + '_mean.npy'):
            tmp = np.load(cfg.dataset.mean_dir + self.experiment_name + '_mean.npy')
            self.num_iteration = tmp[6]
            self.mean1 = tmp[0:3]
            self.mean2 = tmp[3:6]
            logging.info('previous mean of img1 : {}, mean of img2: {}'.format(self.mean1, self.mean2))
        else:
            self.mean1 = np.array([0.35315346, 0.3880523, 0.40808736])
            self.mean2 = np.array([0.35315346, 0.3880523, 0.40808736])
            self.num_iteration = 1
            logging.info('default mean : {}'.format(self.mean1))

        # RNN init state
        if use_rnn:
            self.rnn_hidden_shapes = rnn_hidden_shapes
            self.rnn_staff = [mx.nd.zeros(item[1]) for item in rnn_hidden_shapes]
        else:
            self.rnn_hidden_shapes = []
            self.rnn_staff = []

        # augmentation coeff schedule
        self.half_life = half_life
        self.initial_coeff = initial_coeff
        self.final_coeff = final_coeff
        self.reset()

    @property
    def provide_data(self):
        return [('img1', self.batch_shape), ('img2', self.batch_shape)] + self.rnn_hidden_shapes

    @property
    def provide_label(self):
        """ we assume that names of output and names of corresponding label have the same prefix
            like : 'loss1_output', 'loss1_label'
        """
        return [(item[0].replace('output', 'label'), item[1]) for item in self.label_shape]

    def _thread_start(self):
        # init workers
        self.stop_flag = False
        discount_coeff = self.initial_coeff + \
                         (self.final_coeff - self.initial_coeff) * \
                         (2.0 / (1.0 + exp(-1.0*self.num_iteration/self.half_life)) - 1.0)
        self.worker_proc = [mp.Process(target=numpyloader._worker,
                                       args=[self.data_queue,
                                             self.result_queue,
                                             self.stop_word,
                                             self.stop_flag,
                                             self.get_data,
                                             self.augmentation,
                                             self.data_type,
                                             self.label_shape,
                                             self.interpolation_method,
                                             discount_coeff])
                            for pid in range(self.n_thread)]
        for item in self.worker_proc:
            item.daemon = True
        [item.start() for item in self.worker_proc]

        def cleanup():
            self._shutdown()
        atexit.register(cleanup)

    @staticmethod
    def _worker(data_queue, result_queue, stop_word, stop_flag, get_data, augmentation,
                data_type, label_shape, interpolation_method, discount_coeff):
        count = 0
        for item in iter(data_queue.get, stop_word):
            if stop_flag:
                break
            count += 1
            img1, img2, label, index = get_data(item)
            label = label.astype(np.float64)
            img1, img2, label = augmentation(img1, img2, label, discount_coeff)
            img1 *= 0.00392156862745098
            img2 *= 0.00392156862745098
            labels = []
            for item in label_shape:
                label_resized = resize(label, data_type, interpolation_method, item[1][2], item[1][3])
                labels.append(label_resized)
            result_queue.put((img1, img2, labels, index))

    def _insert_queue(self):

        for item in self.data_dirs:
            self.data_queue.put(item)
        [self.data_queue.put(self.stop_word) for pid in range(self.n_thread)]

    def _save_mean(self):
        # save mean and num of iterations
        logging.info('mean1: {}, mean2: {}'.format(self.mean1, self.mean2))
        logging.info('number of iteration: {}'.format(self.num_iteration))
        tmp = np.r_[self.mean1, self.mean2, self.num_iteration]
        np.save(cfg.dataset.mean_dir + self.experiment_name + '_mean.npy', tmp)

    def iter_next(self):

        if self.current_index >= self.data_num - self.pad:
            self._save_mean()
            return False

        self.first_img = []
        self.second_img = []
        self.auxs = []
        self.label = [[] for i in range(len(self.label_shape))]
        tot_mean1 = np.zeros(3)
        tot_mean2 = np.zeros(3)

        for i in range(self.current_index, self.current_index+self.batch_shape[0]):

            img1, img2, label, aux = self.result_queue.get()

            tot_mean1 += img1.reshape(-1, 3).mean(axis=0)
            tot_mean2 += img2.reshape(-1, 3).mean(axis=0)

            for j in range(len(self.label)):
                self.label[j].append(label[j])

            self.first_img.append(img1)
            self.second_img.append(img2)
            self.auxs.append(aux)

        # update mean
        self.mean1 = self.mean1 * self.num_iteration * self.batch_shape[0] + tot_mean1
        self.mean2 = self.mean2 * self.num_iteration * self.batch_shape[0] + tot_mean2
        self.num_iteration += 1
        self.mean1 /= self.num_iteration * self.batch_shape[0]
        self.mean2 /= self.num_iteration * self.batch_shape[0]

        self.first_img = np.array(self.first_img) - self.mean1
        self.second_img = np.array(self.second_img) - self.mean2

        self.current_index += self.batch_shape[0]
        return True

    def getdata(self):
        return [mx.nd.array(self.first_img.transpose(0, 3, 1, 2)),
                mx.nd.array(self.second_img.transpose(0, 3, 1, 2))] + self.rnn_staff

    @property
    def getaux(self):
        return np.array(self.auxs)

    def getlabel(self):
        if self.data_type =='stereo':
            return [mx.nd.array(np.expand_dims(np.array(self.label[i]), 1)) for i in range(len(self.label))]
        elif self.data_type == 'flow':
            return [mx.nd.array(np.array(self.label[i]).transpose(0, 3, 1, 2)) for i in range(len(self.label))]

    def getindex(self):
        return self.current_index

    def reset(self):
        self.current_index = 0
        if self.be_shuffle:
            shuffle(self.data_dirs)
        self._shutdown()
        self._insert_queue()
        self._thread_start()

    def _shutdown(self):
        # shutdown multi-process
        while True:
            try:
                self.result_queue.get(timeout=1)
            except Queue.Empty:
                break
        while True:
            try:
                self.data_queue.get(timeout=1)
            except Queue.Empty:
                break
        self.stop_flag = True
        if self.worker_proc:
            for i, worker in enumerate(self.worker_proc):
                worker.join(timeout=1)
                if worker.is_alive():
                    logging.error('worker {} is join fail'.format(i))
                    worker.terminate()
