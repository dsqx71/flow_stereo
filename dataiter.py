import atexit
import logging
import multiprocessing as mp
from collections import namedtuple
from random import randint, uniform

import Queue
import caffe
import cv2
import mxnet as mx
import numpy as np
from sklearn import utils

from config import cfg
import util_cython

DataBatch = namedtuple('DataBatch', ['data', 'label', 'pad', 'index'])
class Dataiter_training(mx.io.DataIter):

    def __init__(self, dataset, batch_shape, label_shape, augment_ratio, n_thread=40, be_shuffle=True,
                 downsample_method = 'interpolate',use_rnn=False):

        super(Dataiter_training, self).__init__()

        self.batch_size = batch_shape[0]
        self.batch_shape = batch_shape
        self.shapes = batch_shape[1:]
        self.be_shuffle = be_shuffle
        self.downsample_method = downsample_method

        self.label_shape = label_shape
        self.data_dirs = utils.shuffle(dataset.dirs) if self.be_shuffle else dataset.dirs
        self.num_imgs = len(self.data_dirs)
        self.get_data_function = dataset.get_data
        self.current = 0
        self.data_type = dataset.data_type
        self.use_rnn = use_rnn
        self.augment_ratio = augment_ratio

        # setting of multi-process
        self.stop_word = '==STOP--'
        self.n_thread = n_thread

        self.worker_proc = None
        self.stop_flag = mp.Value('b', False)
        self.result_queue = mp.Queue(maxsize=self.batch_size*20)
        self.data_queue = mp.Queue()
        self.dim_out = 1 if self.data_type == 'stereo' else 2

    @property
    def provide_data(self):

        if self.use_rnn:
            return [('img1', self.batch_shape), ('img2', self.batch_shape), ('init_h',(self.batch_size,cfg.RNN.num_hidden))]
        else:
            return [('img1', self.batch_shape), ('img2', self.batch_shape)]


    @property
    def provide_label(self):

        return [('{}_downsample{}'.format(self.data_type, i+1),
                (self.batch_size, self.dim_out, self.label_shape[i][0], self.label_shape[i][1])) for i in range(len(self.label_shape))]

    def _thread_start(self):

        # init workers
        self.stop_flag = False
        self.worker_proc = [mp.Process(target=Dataiter_training._worker,
                                       args=[pid,
                                             self.data_queue,
                                             self.result_queue,
                                             self.stop_word,
                                             self.stop_flag,
                                             self.get_data_function,
                                             get_augment,
                                             self.augment_ratio,
                                             self.data_type,
                                             crop_or_pad,
                                             self.shapes,
                                             self.downsample_method,
                                             self.label_shape])
                            for pid in range(self.n_thread)]
        [item.start() for item in self.worker_proc]

        def cleanup():
            self.shutdown()
        atexit.register(cleanup)

    def _insert_queue(self):
        for item in self.data_dirs:
            self.data_queue.put(item)
        [self.data_queue.put(self.stop_word) for pid in range(self.n_thread)]

    def iter_next(self):

        if self.current >= self.num_imgs :
            return False

        self.first_img = []
        self.second_img = []
        self.auxs = []
        self.label = [[] for i in range(len(self.label_shape))]

        for i in range(self.current, self.current+self.batch_size):

            img1, img2, label, aux = self.result_queue.get()
            img1 = img1 - img1.reshape(-1, 3).mean(axis=0)
            img2 = img2 - img2.reshape(-1, 3).mean(axis=0)
            for j in range(len(self.label)):
                self.label[j].append(label[j])

            self.first_img.append(img1)
            self.second_img.append(img2)
            self.auxs.append(aux)

        self.current += self.batch_size
        return True

    @staticmethod
    def _worker(worker_id, data_queue, result_queue, stop_word, stop_flag, get_data_function, get_augment,
                augment_ratio, data_type, crop_or_pad, shapes, downsample_method, label_shape):

        for item in iter(data_queue.get, stop_word):

            if stop_flag == 1:
                break

            img1, img2, label, index = get_data_function(item, data_type)
            label = label.astype(np.float64)

            img1 = img1 * 0.00392156862745098
            img2 = img2 * 0.00392156862745098

            if uniform(0, 1) < augment_ratio:
                img1, img2, label = get_augment(img1, img2, label, data_type)
            img1, img2, label = crop_or_pad(img1, img2, label, shapes, True, data_type)

            labels = []
            for j in range(len(label_shape)):
                if downsample_method == 'interpolate':
                    # the interploate method will consider NaN_point ratio
                    if data_type == 'stereo':
                        labels.append(util_cython.resize(label, label_shape[j][1], label_shape[j][0], 0.5))
                    else:
                        tmp = np.zeros(label_shape[j] + (2,))
                        for i in range(2):
                            tmp[:,:,i] = util_cython.resize(label[:, :, i], label_shape[j][1], label_shape[j][0], 0.3)
                        tmp = tmp.transpose(2,0,1)
                        labels.append(tmp)

                else:
                    # Tusimple lidar data is too sparse to interpolate
                    factor = int(label.shape[0]/label_shape[j][0])
                    labels.append(label[::factor,::factor])
            result_queue.put((img1, img2, labels, index))

    def getdata(self):
        return [mx.nd.array(np.asarray(self.first_img).transpose(0, 3, 1, 2)),
                mx.nd.array(np.asarray(self.second_img).transpose(0, 3, 1, 2))]

    @property
    def getaux(self):
        return np.asarray(self.auxs)

    def getlabel(self):

        if self.data_type =='stereo':
            return [mx.nd.array(np.expand_dims(np.asarray(self.label[i]), 1)) for i in range(len(self.label_shape))]

        elif self.data_type == 'flow':
            return [mx.nd.array(np.asarray(self.label[i])) for i in range(len(self.label_shape))]

    def getindex(self):

        return self.current

    def reset(self):

        self.current = 0
        if self.be_shuffle:
            self.data_dirs = utils.shuffle(self.data_dirs)

        self.shutdown()
        self._insert_queue()
        self._thread_start()

    def shutdown(self):
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

class multi_imageRecord(mx.io.DataIter):

    def __init__(self, records, data_type, batch_shape, label_shapes, augment_ratio,use_rnn):

        super(multi_imageRecord, self).__init__()
        self.data_type = data_type
        self.records = records
        self.augment_ratio = augment_ratio
        self.batch_size = batch_shape[0]
        self.shapes = batch_shape[1:]
        self.label_shapes = label_shapes
        self.reset()
        self.use_rnn = use_rnn
        if use_rnn:
            self.provide_data = [('img1', batch_shape),('img2', batch_shape), ('init_h',(batch_shape[0],cfg.RNN.num_hidden))]
        else:
            self.provide_data = [('img1', batch_shape), ('img2', batch_shape)]
        if self.data_type == 'stereo':
            self.provide_label = [('{}_downsample{}'.format(self.data_type,j+1),(self.batch_size,1)+label_shapes[j]) \
                                  for j in xrange(len(label_shapes))]
        elif self.data_type == 'flow':
            self.provide_label = [('{}_downsample{}'.format(self.data_type, j+1),(self.batch_size,2) + label_shapes[j])\
                                  for j in xrange(len(label_shapes))]
    def iter_next(self):

        data1 = self.records[0].next().data[0]
        data2 = self.records[1].next().data[0]
        if len(self.records) == 3:
            labels = self.records[2].next().data[0].asnumpy().sum(1)
        else:
            labels = None

        tmp1 = data1.asnumpy().transpose(0, 2, 3, 1)
        tmp2 = data2.asnumpy().transpose(0, 2, 3, 1)
        self.img1 = []
        self.img2 = []

        self.labels = [[] for i in xrange(len(self.label_shapes))]

        for j in xrange(self.batch_size):

            img1, img2, label = crop_or_pad(tmp1[j], tmp2[j], labels[j], self.shapes, True, self.data_type)

            img1 = img1 * 0.00392156862745098
            img2 = img2 * 0.00392156862745098

            img1 = img1 - img1.reshape(-1, 3).mean(axis=0)
            img2 = img2 - img2.reshape(-1, 3).mean(axis=0)

            self.img1.append(img1)
            self.img2.append(img2)
            # if self.data_type == 'stereo':
            #     label[label <= 0] = np.nan
            label = label.astype(np.float64)
            for i in xrange(len(self.labels)):
                self.labels[i].append(util_cython.resize(label, self.label_shapes[i][1], self.label_shapes[i][0]))

        return True

    def getdata(self):
        return [mx.nd.array(np.asarray(self.img1).transpose(0, 3, 1, 2)),
                mx.nd.array(np.asarray(self.img2).transpose(0, 3, 1, 2))]

    def getlabel(self):
        #  output dimension of optical flow is 3,and stereo is 2

        if len(self.labels[0][0].shape) == 3:
            return [mx.nd.array(np.asarray(self.labels[i]).transpose(0, 3, 1, 2)) for i in xrange(len(self.labels))]
        else:
            return [mx.nd.array(np.expand_dims(np.asarray(self.labels[i]),3).transpose(0, 3, 1, 2))
                    for i in xrange(len(self.labels))]

    def getindex(self):
        return self.records[0].getindex()

    def reset(self):
        for i in self.records:
            i.reset()

class caffe_iterator(mx.io.DataIter):

    def __init__(self, ctx, dataset, batch_shape, label_shape, input_shape,
                 template = cfg.dataset.prototxt_template,
                 caffe_prototxt = cfg.dataset.prototxt_dir,
                 caffe_pretrain = cfg.dataset.pretrain_caffe,
                 n_thread=15, be_shuffle=True, augment=True,use_rnn=False):


        super(caffe_iterator, self).__init__()
        self.be_shuffle = be_shuffle
        self.label_shape = label_shape
        self.input_shape = input_shape
        self.data_dirs = utils.shuffle(dataset.dirs) if self.be_shuffle else dataset.dirs

        self.num_imgs = len(self.data_dirs)
        self.batch_shape = batch_shape
        self.batch_size = batch_shape[0]
        self.ctx = ctx
        self.use_rnn = use_rnn
        self.get_data_function = dataset.get_data

        # change prototxt setting
        replacement_list = \
            {
            '$batchsize': ('%d' % batch_shape[0]),
            '$input_height' : ('%d' %self.input_shape[0]),
            '$input_width': ('%d' % self.input_shape[1]),
            '$crop_height': ('%d' % self.batch_shape[2]),
            '$crop_width': ('%d' % self.batch_shape[3]),
            }

        with open(template, "r") as tfile:
            ori_proto = tfile.read()

        for r in replacement_list:
            ori_proto = ori_proto.replace(r, replacement_list[r])

        with open(caffe_prototxt, "w") as tfile:
            tfile.write(ori_proto)

        for ctx in self.ctx:
            caffe.set_device(ctx.device_id)

        caffe.set_mode_gpu()
        self.caffe_net = caffe.Net(caffe_prototxt, caffe_pretrain, caffe.TRAIN if augment else caffe.TEST)

        with open(caffe_prototxt, "w") as tfile:
            tfile.write(ori_proto)

        self.stop_word = '==STOP--'
        self.n_thread = n_thread
        self.worker_proc = None
        self.stop_flag = mp.Value('b', False)
        self.result_queue = mp.Queue(maxsize=self.batch_shape[0] * 20)
        self.data_queue = mp.Queue()


    def _thread_start(self):
        # init workers
        self.stop_flag = False
        self.worker_proc = [mp.Process(target=caffe_iterator._worker,
                                       args=[pid,
                                             self.data_queue,
                                             self.result_queue,
                                             self.stop_word,
                                             self.stop_flag,
                                             self.get_data_function,
                                             self.input_shape])
                            for pid in range(self.n_thread)]
        [item.start() for item in self.worker_proc]

        def cleanup():
            self.shutdown()

        atexit.register(cleanup)

    def _insert_queue(self):
        for item in self.data_dirs:
            self.data_queue.put(item)
        [self.data_queue.put(self.stop_word) for pid in range(self.n_thread)]

    @staticmethod
    def _worker(worker_id, data_queue, result_queue, stop_word, stop_flag, get_data_function, input_shape):

        for item in iter(data_queue.get, stop_word):
            if stop_flag == 1:
                break
            img1, img2, label, index = get_data_function(item, 'stereo')
            img1, img2, label = crop_or_pad(img1, img2, label, (3,) + input_shape, True, 'stereo')
            result_queue.put((img1, img2, label, index))

    @property
    def provide_data(self):
        if self.use_rnn:
            return [('img1', self.batch_shape), ('img2', self.batch_shape),('init_h',(self.batch_shape[0],cfg.RNN.num_hidden))]
        else:
            return [('img1', self.batch_shape), ('img2', self.batch_shape)]

    @property
    def provide_label(self):
        return [('stereo_downsample%d' % (i+1),(self.batch_shape[0],1) + self.label_shape[i])
                for i in range(len(self.label_shape))]

    def iter_next(self):

        if self.current + self.batch_size > self.num_imgs:
            return False

        img1_list = []
        img2_list = []
        label_list = []
        self.label = []
        self.aux_list = []

        for i in range(self.current, self.current + self.batch_size):
            img1, img2, label, aux = self.result_queue.get()
            img1_list.append(img1.transpose(2,0,1))
            img2_list.append(img2.transpose(2,0,1))
            label_list.append(label)
            self.aux_list.append(aux)

        img1_list = np.array(img1_list)
        img2_list = np.array(img2_list)
        label_list = np.expand_dims(np.array(label_list),1)

        self.caffe_net.blobs['blob0'].data[:] = img1_list
        self.caffe_net.blobs['blob1'].data[:] = img2_list
        self.caffe_net.blobs['blob2'].data[:] = label_list
        self.caffe_net.forward()

        for i in range(len(self.label_shape)):

            self.label.append([])
            for j in range(self.batch_size):
                self.label[i].append(
                    util_cython.resize(self.caffe_net.blobs['disp_gt_aug'].data[j, 0].astype(np.float64),
                                       self.label_shape[i][1],
                                       self.label_shape[i][0]))
            self.label[i] = mx.nd.array(np.expand_dims(np.array(self.label[i]),1))

        self.current += self.batch_size

        return True

    def getdata(self):

        return [mx.nd.array(self.caffe_net.blobs['img0_aug'].data.copy()),
                mx.nd.array(self.caffe_net.blobs['img1_aug'].data.copy())]

    def getlabel(self):

        return self.label

    @property
    def getaux(self):

        return np.asarray(self.auxs)

    def getindex(self):

        return self.current

    def reset(self):

        self.current = 0

        if self.be_shuffle:
            self.data_dirs = utils.shuffle(self.data_dirs)
        # print self.data_dirs
        self.shutdown()
        self._insert_queue()
        self._thread_start()

    def shutdown(self):

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


def crop_or_pad(img1, img2, label, shapes, is_train, data_type):

    y_ori, x_ori = img1.shape[:2]
    y, x = shapes[1:]
    if x == x_ori and y == y_ori:
        return img1, img2, label

    elif y >= y_ori and x >= x_ori:

        # padding
        tmp1 = np.zeros((y, x, 3))
        tmp2 = np.zeros((y, x, 3))
        if is_train:
            x_begin = randint(0, x-x_ori)
            y_begin = randint(0, y-y_ori)
        else:
            x_begin = 0
            y_begin = 0
        tmp1[y_begin:y_begin+y_ori, x_begin:x_begin + x_ori, :] = img1[:]
        tmp2[y_begin:y_begin+y_ori, x_begin:x_begin + x_ori, :] = img2[:]
        if label is not None:
            if data_type == 'stereo':
                tmp3 = np.zeros((y, x))
                tmp3[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori] = label[:]
            else:
                tmp3 = np.zeros((y, x, 2))
                tmp3[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori,:] = label[:]
            return tmp1, tmp2, tmp3
        else:
            return tmp1, tmp2, None
    elif y<=y_ori and x <= x_ori:
        # cropping
        x_begin = randint(0, x_ori - x )
        y_begin = randint(0, y_ori - y )
        if label is not None:
            return img1[y_begin:y_begin+y, x_begin:x_begin+x, :], img2[y_begin:y_begin+y, x_begin:x_begin+x, :],\
                label[y_begin:y_begin+y, x_begin:x_begin+x]
        else:
            return img1[y_begin:y_begin+y, x_begin:x_begin+x, :], img2[y_begin:y_begin+y, x_begin:x_begin+x, :], None

def get_augment(img1, img2, label, data_type):

    rows, cols, _ = img1.shape
    rotate_range = cfg.dataset.rotate_range
    translation_range = cfg.dataset.translation_range
    gaussian_noise = cfg.dataset.gaussian_noise

    rgb_cof = np.random.uniform(low=cfg.dataset.rgbmul[0], high=cfg.dataset.rgbmul[1], size=3)
    gaussian_noise_scale = uniform(0.0, gaussian_noise)
    rotate = randint(-rotate_range, rotate_range)
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate, 1)

    tx = randint(int(-img1.shape[1] * translation_range), int(img1.shape[1] * translation_range))
    ty = randint(int(-img1.shape[0] * translation_range), int(img1.shape[0] * translation_range))
    M = np.float32([[1, 0, tx], [0, 1, ty]])

    beta = uniform(cfg.dataset.beta[0], cfg.dataset.beta[1])
    alpha = uniform(cfg.dataset.alpha[0], cfg.dataset.alpha[1])

    # multiply rgb factor plus guassian noise
    img1 = img1 * rgb_cof + np.random.normal(loc=0.0, scale=gaussian_noise_scale,size=img1.shape)
    img2 = img2 * rgb_cof + np.random.normal(loc=0.0, scale=gaussian_noise_scale,size=img1.shape)

    # rotation
    if data_type != 'stereo':
        img1 = cv2.warpAffine(img1, rotation_matrix, (cols, rows))
        img2 = cv2.warpAffine(img2, rotation_matrix, (cols, rows))
        label = cv2.warpAffine(label, rotation_matrix, (cols, rows))

    # translation
    # img1 = cv2.warpAffine(img1, M, (cols, rows))
    # img2 = cv2.warpAffine(img2, M, (cols, rows))
    # label = cv2.warpAffine(label, M, (cols, rows))
    # label[label==0] = np.nan

    # brightness and contrastness
    img1 = cv2.multiply(img1, np.array([alpha]))
    img1 = cv2.add(img1, np.array([beta]))
    img2 = cv2.multiply(img2, np.array([alpha]))
    img2 = cv2.add(img2, np.array([beta]))

    # # scaling
    # origin_shape = img1.shape[:2]
    # if uniform(0, 1) < 0.5:
    #     y_shape = randint(1.0*origin_shape[0],int(cfg.dataset.scale[1]*origin_shape[0]))
    #     x_shape = randint(1.0*origin_shape[1],int(cfg.dataset.scale[1]*origin_shape[1]))
    # else:
    #     y_shape = randint(int(cfg.dataset.scale[0] * origin_shape[0]), 1.0 * origin_shape[0])
    #     x_shape = randint(int(cfg.dataset.scale[0] * origin_shape[1]), 1.0 * origin_shape[1])
    #
    # img1 = cv2.resize(img1, (x_shape, y_shape))
    # img2 = cv2.resize(img2, (x_shape, y_shape))
    #
    # if data_type == 'stereo':
    #     label = label * (float(x_shape)/origin_shape[1])
    # else:
    #     label[:, :, 0] = label[:, :, 0] * (float(x_shape)/origin_shape[1])
    #     label[:, :, 1] = label[:, :, 1] * (float(y_shape)/origin_shape[0])
    # label = label.astype(np.float64)
    # label = util_cython.resize(label,x_shape,y_shape)
    # label[label==0] = np.nan
    if  uniform(0,1) < cfg.dataset.flip_rate:

        img1 = cv2.flip(img1,0)
        img2 = cv2.flip(img2,0)
        label = cv2.flip(label,0)

    return img1, img2, label