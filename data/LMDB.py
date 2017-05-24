import logging
import atexit
import os
import Queue
import mxnet as mx
import numpy as np
import lmdb
import caffe_datum
import multiprocessing as mp

from collections import namedtuple
from math import exp
from .config import cfg
from .data_util import resize
from random import shuffle

import time

DataBatch = namedtuple('DataBatch', ['data', 'label', 'pad', 'index'])
class lmdbloader(mx.io.DataIter):

    def __init__(self,
                 lmdb_path,
                 data_type,
                 ctx,
                 experiment_name,
                 augmentation,
                 batch_shape,
                 label_shape,
                 n_thread=15,
                 interpolation_method='nearest',
                 use_rnn=False,
                 rnn_hidden_shapes=None,
                 initial_coeff=0.1,
                 final_coeff=1.0,
                 half_life=50000,
                 chunk_size=16):

        super(lmdbloader, self).__init__()

        # Set up LMDB
        lmdb_env = lmdb.open(lmdb_path)
        self.lmdb_txn = lmdb_env.begin()
        self.datum = caffe_datum.Datum()

        # ctx
        self.ctx = ctx[0]

        # shapes
        self.batch_shape = batch_shape # (batchsize, channel, height, width)
        self.target_shape = batch_shape[2:] # target_height, target_width of input
        self.label_shape = label_shape # dict of label names and label shapes

        # preprocessing
        self.interpolation_method = interpolation_method
        self.augmentation = augmentation
        assert 'bilinear' == interpolation_method or 'nearest' == interpolation_method, 'wrong interpolation method'

        # setting of data
        self.data_type = data_type
        self.data_num = self.lmdb_txn.stat()['entries']
        self.current_index = 0
        self.pad = self.data_num % self.batch_shape[0]
        self.experiment_name = experiment_name
        assert self.data_type == 'stereo' or self.data_type == 'flow', 'wrong data type'

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
            self.rnn_stuff = [mx.nd.zeros(item[1]) for item in rnn_hidden_shapes]
        else:
            self.rnn_hidden_shapes = []
            self.rnn_stuff = []

        # augmentation coeff schedule
        self.half_life = half_life
        self.initial_coeff = initial_coeff
        self.final_coeff = final_coeff

        # data chunk
        self.chunk_size = chunk_size

        # setting of multi-process
        self.stop_word = '==STOP--'
        self.n_thread = n_thread
        self.worker_proc = None
        self.stop_flag = mp.Value('b', False)
        self.result_queue = mp.Queue(maxsize=self.batch_shape[0] * 50)
        self.data_queue = mp.Queue(maxsize=self.batch_shape[0] * 50)
        self.reset()

    @property
    def provide_data(self):
        return [('img1', self.batch_shape), ('img2', self.batch_shape)] + self.rnn_hidden_shapes

    @property
    def provide_label(self):
        """ we assume that names of output and names of corresponding label have the same prefix
            like : 'loss1_output', 'loss1_label'
        """
        return [(item[0], item[1]) for item in self.label_shape]

    def _thread_start(self):
        # init workers
        self.stop_flag = False
        discount_coeff = self.initial_coeff + \
                         (self.final_coeff - self.initial_coeff) * \
                         (2.0 / (1.0 + exp(-1.0*self.num_iteration/self.half_life)) - 1.0)
        logging.info("discount coeff: {}".format(discount_coeff))

        # consumer
        self.worker_proc = [mp.Process(target=lmdbloader._consumer,
                                       args=[self.data_queue,
                                             self.result_queue,
                                             self.stop_word,
                                             self.stop_flag,
                                             self.augmentation,
                                             self.data_type,
                                             self.label_shape,
                                             self.interpolation_method,
                                             discount_coeff,
                                             self.datum])
                            for _ in range(self.n_thread)]
        # producer
        self.worker_proc.append(mp.Process(target=lmdbloader._producer,
                                           args=[self.data_queue,
                                                 self.lmdb_txn.cursor().iternext(),
                                                 self.stop_word,
                                                 self.chunk_size,
                                                 self.n_thread]))
        for item in self.worker_proc:
            item.daemon = True
        [item.start() for item in self.worker_proc]

        def cleanup():
            self._shutdown()
        atexit.register(cleanup)

    def _save_mean(self):
        # save mean and num of iterations
        logging.info('mean1: {}, mean2: {}'.format(self.mean1, self.mean2))
        logging.info('number of iteration: {}'.format(self.num_iteration))
        tmp = np.r_[self.mean1, self.mean2, self.num_iteration]
        np.save(cfg.dataset.mean_dir + self.experiment_name + '_mean.npy', tmp)

    @staticmethod
    def _consumer(data_queue, result_queue, stop_word, stop_flag, augmentation,
                data_type, label_shape, interpolation_method, discount_coeff, datum):
        count = 0
        for key, value in iter(data_queue.get, stop_word):
            if stop_flag:
                break
            count += 1
            datum.ParseFromString(value)
            data = np.fromstring(datum.data[:6*datum.height*datum.width], dtype=np.uint8)
            label = np.fromstring(datum.data[6*datum.height*datum.width:], dtype=np.int16)
            data = data.reshape(6, datum.height, datum.width).transpose(1, 2, 0)
            img1 = data[..., :3]
            img2 = data[..., 3:]
            label = label / -32.0

            if data_type == 'stereo':
                label = label.reshape(datum.height, datum.width)
            else:
                label = label * -1
                label = label[:2*datum.height*datum.width]
                label = label.reshape(2, datum.height, datum.width).transpose(1, 2, 0)

            label = label.astype(np.float64)
            img1, img2, label = augmentation(img1, img2, label, discount_coeff)

            img1 = img1 * 0.00392156862745098
            img2 = img2 * 0.00392156862745098


            labels = []
            for item in label_shape:
                label_resized = resize(label, data_type, interpolation_method, item[1][2], item[1][3])
                label_resized = np.round(label_resized).astype(np.int16)
                labels.append(label_resized)

            mean1 = img1.reshape(-1, 3).mean(axis=0)
            mean2 = img2.reshape(-1, 3).mean(axis=0)
            img1 = img1.astype(np.float16)
            img2 = img2.astype(np.float16)

            result_queue.put((img1, img2, labels, key, mean1, mean2))

    @staticmethod
    def _producer(data_queue, lmdb_iter, stop_word, chunk_size, n_thread):

        chunk = []
        while True:
            try:
                item = lmdb_iter.next()
                chunk.append(item)
                if len(chunk) == chunk_size:
                    shuffle(chunk)
                    data_queue.put(chunk.pop())
            except StopIteration:
                break
        # move remnant data to data_queue
        for item in chunk:
            data_queue.put(item)

        for i in range(n_thread):
            data_queue.put(stop_word)

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
            img1, img2, label, aux, mean1, mean2 = self.result_queue.get()
            tot_mean1 += mean1
            tot_mean2 += mean2

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

        self.first_img = mx.nd.array(self.first_img, ctx=self.ctx) - mx.nd.array(self.mean1, ctx=self.ctx)
        self.second_img = mx.nd.array(self.second_img, ctx=self.ctx) - mx.nd.array(self.mean2, ctx=self.ctx)
        self.current_index += self.batch_shape[0]

        return True

    def getdata(self):
        return [mx.nd.transpose(self.first_img, axes=(0, 3, 1, 2)),
                mx.nd.transpose(self.second_img, axes=(0, 3, 1, 2))] + self.rnn_stuff

    @property
    def getaux(self):
        return np.array(self.auxs)

    def getlabel(self):
        if self.data_type =='stereo':
            return [mx.nd.expand_dims(mx.nd.array(self.label[i], ctx=self.ctx), axis=1) for i in range(len(self.label))]
        elif self.data_type == 'flow':
            return [mx.nd.transpose(mx.nd.array(self.label[i], ctx=self.ctx), axes=(0, 3, 1, 2)) for i in range(len(self.label))]

    def getindex(self):
        return self.current_index

    def reset(self):

        self.current_index = 0
        self._shutdown()
        self._thread_start()






