import mxnet as mx
import numpy as np
import cv2
import atexit
import logging
import multiprocessing as mp
import Queue
import time
from collections import namedtuple
from random import shuffle
from math import exp

# radius of patches
s1 = 6
s2 = 13

DataBatch = namedtuple('DataBatch', ['data', 'label', 'pad', 'index'])
class patchiter(mx.io.DataIter):

    def __init__(self,
                 ctx,
                 experiment_name,
                 dataset,
                 batch_size,
                 img_augmentation,
                 patch_augmentation,
                 low,
                 high,
                 n_thread=40,
                 be_shuffle=True,
                 initial_coeff=0.1,
                 final_coeff=1.0,
                 half_life=50000,
                 num_iteration=10):
        super(patchiter, self).__init__()
        # ctx
        self.ctx = ctx[0]
        self.batch_size = batch_size

        # Setting of data
        self.be_shuffle = be_shuffle
        self.data_dirs  = dataset.dirs
        self.data_num = len(self.data_dirs)
        self.get_data = dataset.get_data
        self.data_type = dataset.data_type
        self.current_index = 0
        self.experiment_name = experiment_name

        # Preprocessing
        self.img_augmentation = img_augmentation
        self.patch_augmentation = patch_augmentation

        # augmentation coeff schedule
        self.half_life = half_life
        self.initial_coeff = initial_coeff
        self.final_coeff = final_coeff
        self.num_iteration = num_iteration
        # Setting of negative mining
        self.low = low
        self.high = high

        # setting of multi-process
        self.stop_word = '==STOP--'
        self.n_thread = n_thread
        self.worker_proc = None
        self.stop_flag = mp.Value('b', False)
        self.result_queue = mp.Queue()
        self.data_queue = mp.Queue(batch_size*30)
        self.input_queue = mp.Queue()
        self.reset()
    @property
    def provide_data(self):
        return [('left', (self.batch_size, 3, 13, 13)),
                ('right', (self.batch_size, 3, 13, 13)),
                ('left_downsample', (self.batch_size, 3, 13, 13)),
                ('right_downsample', (self.batch_size, 3, 13, 13))]

    @property
    def provide_label(self):
        return [('label', (self.batch_size,))]

    def _thread_start(self):
        # init workers
        # producer
        discount_coeff = self.initial_coeff + \
                         (self.final_coeff - self.initial_coeff) * \
                         (2.0 / (1.0 + exp(-1.0*self.num_iteration/self.half_life)) - 1.0)
        logging.info("discount_coeff : {}".format(discount_coeff))
        self.stop_flag = False
        self.worker_proc = [mp.Process(target=patchiter._producer,
                                       args=[
                                           self.input_queue,
                                           self.data_queue,
                                           self.stop_word,
                                           self.get_data,
                                           self.data_type,
                                           self.img_augmentation,
                                           self.patch_augmentation,
                                           self.low,
                                           self.high,
                                           self.stop_flag,
                                           discount_coeff
                                       ])
                            for pid in range(self.n_thread)]

        # consumer
        self.worker_proc.append(mp.Process(target=patchiter._consumer,
                                           args=[self.data_queue,
                                                 self.result_queue,
                                                 self.batch_size,
                                                 self.stop_word,
                                                 self.stop_flag,
                                                 self.n_thread]))
        for item in self.worker_proc:
            item.daemon = True
        [item.start() for item in self.worker_proc]

        def cleanup():
            self._shutdown()
        atexit.register(cleanup)

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
        while True:
            try:
                self.input_queue.get(timeout=1)
            except Queue.Empty:
                break

        self.stop_flag = True
        if self.worker_proc:
            for i, worker in enumerate(self.worker_proc):
                worker.join(timeout=1)
                if worker.is_alive():
                    logging.error('worker {} is join fail'.format(i))
                    worker.terminate()

    def _insert_queue(self):
        for item in self.data_dirs:
            self.input_queue.put(item)
        [self.input_queue.put(self.stop_word) for pid in range(self.n_thread)]

    @staticmethod
    def _producer(input_queue, data_queue, stop_word, get_data, data_type, img_augmentation,
                  patch_augmentation, low, high, stop_flag, discount_coeff):
        """
        Refer to Zhuoyuan Chen et al ICCV 2015
        """
        for item in iter(input_queue.get, stop_word):
            # Read data
            if stop_flag:
                break
            img1, img2, label, aux = get_data(item)
            label = np.round(label)
            if img_augmentation is not None:
                img1, img2, label = img_augmentation(img1, img2, label, discount_coeff)
            # Standardize
            img1 = (img1 - img1.reshape(-1, 3).mean(axis=0)) / img1.reshape(-1, 3).std(axis=0)
            img2 = (img2 - img2.reshape(-1, 3).mean(axis=0)) / img2.reshape(-1, 3).std(axis=0)
            # Extract patches
            data = [[] for i in range(5)]
            if data_type == 'stereo':
                y_range = range(s2, img1.shape[0]-s2)
                shuffle(y_range)
                for y in y_range:
                    x_range = range(s2, img1.shape[0]-s2)
                    for x in x_range:
                        if np.isnan(label[y, x]) == False:
                            # disparity
                            dis = int(label[y, x])
                            if x - dis - s2 >= 0:
                                # Get positive samples according to the ground truth disparity
                                # Four patch
                                data[0].append(img1[y-s1:y+1+s1, x-s1:x+1+s1])
                                data[1].append(img2[y-s1:y+1+s1, x-s1-dis:x+1+s1-dis])
                                data[2].append(cv2.resize(img1[y-s2:y+s2, x-s2:x+s2,:], (0, 0), fx=0.5, fy=0.5))
                                data[3].append(cv2.resize(img2[y-s2:y+s2, x-s2-dis:x+s2-dis,:], (0, 0), fx=0.5, fy=0.5))
                                # label
                                data[4].append(1.0)
                                if patch_augmentation is not None:
                                    patch_augmentation(data[0][-1], data[1][-1], data[2][-1], data[3][-1])

                                # Draw negative samples from uniform distribution which centers around the positive samples
                                temp = [x - dis + move for move in range(low, high) if x-dis+move<img1.shape[1]-s2]
                                temp.extend([x - dis - move for move in range(low, high) if x-dis-move>=s2])
                                # xn: X coordinate of the negative sample
                                xn = np.random.choice(temp)

                                # Four patch
                                data[0].append(img1[y-s1:y+1+s1, x-s1: x+1+s1])
                                data[1].append(img2[y-s1:y+1+s1, xn-s1: xn+1+s1])
                                data[2].append(cv2.resize(img1[y-s2: y+s2, x-s2:x+s2], (0, 0), fx=0.5, fy=0.5))
                                data[3].append(cv2.resize(img2[y-s2: y+s2, xn-s2:xn+s2], (0, 0), fx=0.5, fy=0.5))
                                data[4].append(0.0)
                                if patch_augmentation is not None:
                                    patch_augmentation(data[0][-1], data[1][-1], data[2][-1], data[3][-1])
                    # print 'data_size ', sys.getsizeof(data) / 1024.0
                    data_queue.put(data)
                    data = [[] for i in range(5)]
            else:
                raise NotImplementedError('Not supported.')
        data_queue.put(stop_word)

    @staticmethod
    def _consumer(data_queue, result_queue, batch_size, stop_word, stop_flag, num_thread):

        img1_lst = []
        img2_lst = []
        img1_downsample_lst = []
        img2_downsample_lst = []
        label_lst = []
        tot_stop = 0
        while True:
            item = data_queue.get()
            if item == stop_word:
                tot_stop += 1
                if tot_stop == num_thread:
                    result_queue.put(stop_word)
                    break
            else:
                # print 'get_time', time.time() - tic
                if stop_flag:
                    break

                img1, img2, img1_downsample, img2_downsample, label = item
                img1_lst.extend(img1)
                img2_lst.extend(img2)
                img1_downsample_lst.extend(img1_downsample)
                img2_downsample_lst.extend(img2_downsample)
                label_lst.extend(label)

                # Send
                while len(img1_lst) >= batch_size:
                    # print "length of img1_lst", len(img1_lst)
                    if stop_flag:
                        break
                    result_queue.put((np.array(img1_lst[:batch_size]).astype(np.float16).transpose(0, 3, 1, 2),
                                      np.array(img2_lst[:batch_size]).astype(np.float16).transpose(0, 3, 1, 2),
                                      np.array(img1_downsample_lst[:batch_size]).astype(np.float16).transpose(0, 3, 1, 2),
                                      np.array(img2_downsample_lst[:batch_size]).astype(np.float16).transpose(0, 3, 1, 2),
                                      np.array(label_lst[:batch_size]).astype(np.uint8)))
                    # Delete
                    img1_lst = img1_lst[batch_size:]
                    img2_lst = img2_lst[batch_size:]
                    img1_downsample_lst = img1_downsample_lst[batch_size:]
                    img2_downsample_lst = img2_downsample_lst[batch_size:]
                    label_lst = label_lst[batch_size:]

    def iter_next(self):
        tic = time.time()
        item = self.result_queue.get()
        # print ('iter_next get time', time.time() - tic)
        if self.stop_word == item:
            return False

        img1, img2, img1_downsample, img2_downsample, label = item
        self.img1 = mx.nd.array(img1, ctx=self.ctx)
        self.img2 = mx.nd.array(img2, ctx=self.ctx)
        self.img1_downsample = mx.nd.array(img1_downsample, ctx=self.ctx)
        self.img2_downsample = mx.nd.array(img2_downsample, ctx=self.ctx)
        self.label = mx.nd.array(label, ctx=self.ctx)
        self.current_index += self.batch_size

        self.num_iteration += 1
        return True

    def reset(self):
        if self.be_shuffle:
            shuffle(self.data_dirs)
        self.current_index = 0
        self._shutdown()
        self._insert_queue()
        self._thread_start()

    def getindex(self):
        return self.current_index

    def getdata(self):
        return [self.img1, self.img2, self.img1_downsample, self.img2_downsample]

    def getlabel(self):
        return [self.label]


