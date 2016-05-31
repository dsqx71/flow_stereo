from collections import namedtuple
import cv2
import mxnet as mx
import numpy as np
from sklearn import utils
from config import cfg
from random import randint, uniform
import Queue
import multiprocessing as mp
import atexit
import logging
from config import ctx,batchsize

DataBatch = namedtuple('DataBatch', ['data', 'label', 'pad', 'index'])


class Dataiter(mx.io.DataIter):
    def __init__(self, dataset, batch_shape, is_train, label_shapes, augment_ratio,
                 multi_thread=False, n_thread=40, be_shuffle=True, sub_mean=True, ctx=ctx[0], label_sparse=False):
        """
            data iteraotr of dispnet and flownet
        """
        super(Dataiter, self).__init__()

        # setting of data iterator
        self.batch_size = batch_shape[0]
        self.shapes = batch_shape[1:]
        self.be_shuffle = be_shuffle
        self.ctx = ctx
        self.label_sparse = label_sparse
        if self.be_shuffle:
            self.data_dirs = utils.shuffle(dataset.dirs)
        else:
            self.data_dirs = dataset.dirs
        self.num_imgs = len(self.data_dirs)
        self.is_train = is_train
        self.label_shapes = label_shapes
        self.get_data_function = dataset.get_data
        self.current = 0
        self.data_type = dataset.data_type
        if self.is_train:
            self.augment_ratio = augment_ratio
        else:
            self.augment_ratio = 0
        self.sub_mean = sub_mean

        # setting of multi-process
        self.multi_thread = multi_thread
        self.stop_word = '==STOP--'
        self.n_thread = n_thread
        if self.multi_thread:
            self.worker_proc = None
            self.stop_flag = mp.Value('b', False)
            self.result_queue = mp.Queue(maxsize=self.batch_size*20)
            self.data_queue = mp.Queue()

        # data name and shape
        self.provide_data = [('img1', batch_shape), ('img2', batch_shape)]
        if self.data_type == 'stereo':
            self.provide_label = [ ('{}_downsample{}'.format(self.data_type, j+1), (self.batch_size,1)+label_shapes[j])\
                                   for j in xrange(len(label_shapes))]
        else:
            self.provide_label = [('{}_downsample{}'.format(self.data_type, j+1),(self.batch_size,2) + label_shapes[j])\
                                   for j in xrange(len(label_shapes))]

    def _thread_start(self):
        # init workers
        self.stop_flag = False
        self.worker_proc = [mp.Process(target=Dataiter._worker,
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
                                             self.is_train,
                                             self.shapes,
                                             self.label_shapes])
                            for pid in xrange(self.n_thread)]
        [item.start() for item in self.worker_proc]

        def cleanup():
            self.shutdown()
        atexit.register(cleanup)

    def _insert_queue(self):
        # insert data's directory into queue
        for item in self.data_dirs:
            self.data_queue.put(item)
        [self.data_queue.put(self.stop_word) for pid in xrange(self.n_thread)]

    def iter_next(self):

        if self.current + self.batch_size > self.num_imgs:
            return False
        self.first_img = []
        self.second_img = []
        self.auxs = []
        if self.is_train:
            self.label = [[] for i in xrange(len(self.label_shapes))]
        else:
            self.label = []

        for i in xrange(self.current, self.current+self.batch_size):
            if self.multi_thread:
                if self.is_train:
                    img1, img2, label, aux = self.result_queue.get()
                    for j in xrange(len(self.label)):
                        self.label[j].append(label[j])
                else:
                    img1, img2, label, aux = self.result_queue.get()
                    self.label.append(label)
            else:
                if self.is_train:
                    img1, img2, label, aux = self.get_data_function(self.data_dirs[i], self.data_type)
                    img1, img2, label = crop_or_pad(img1, img2, label, self.shapes, self.is_train, self.data_type)
                    if uniform(0, 1) < self.augment_ratio:
                        img1, img2, label = get_augment(img1, img2, label, self.data_type)
                    label = label.astype(np.float32)
                    for j in xrange(len(self.label)):
                        self.label[j].append(cv2.resize(label, (self.label_shapes[j][1], self.label_shapes[j][0])))
                else:
                    # privide original label,but we must resize the image to match the input of the network.
                    img1, img2, label, aux = self.get_data_function(self.data_dirs[i], self.data_type)
                    img1 = cv2.resize(img1, (self.shapes[2],self.shapes[1]))
                    img2 = cv2.resize(img2, (self.shapes[2],self.shapes[1]))

                    self.label.append(label)

            if self.sub_mean:
                img1 = (img1 * 0.0039216) - np.array([0.411451, 0.432060, 0.450141])
                img2 = (img2 * 0.0039216) - np.array([0.410602, 0.431021, 0.448553])
            self.first_img.append(img1)
            self.second_img.append(img2)
            self.auxs.append(aux)
        self.current += self.batch_size
        return True

    def getdata(self):
        return [mx.nd.array(np.asarray(self.first_img).transpose(0, 3, 1, 2), self.ctx),
                mx.nd.array(np.asarray(self.second_img).transpose(0, 3, 1, 2), self.ctx)]
    @property
    def getaux(self):
        return np.asarray(self.auxs)

    def getlabel(self):
        # output dimension of optical flow is 3,and stereo is 2
        # when training , we provide downsampled labels,but when testing,we only provide origin lable
        if self.is_train:
            if len(self.label[0][0].shape) == 3:
                return [mx.nd.array(np.asarray(self.label[i]).transpose(0, 3, 1, 2), self.ctx) for i in xrange(len(self.label))]
            else:
                return [mx.nd.array(np.expand_dims(np.asarray(self.label[i]), 1), self.ctx) for i in xrange(len(self.label))]
        else:
            if self.label[0] is None:
                return None
            # when testing,it will only provide original labels
            if len(self.label[0].shape) == 3:
                return [mx.nd.array(np.asarray(self.label).transpose(0, 3, 1, 2), self.ctx)]
            else:
                return [mx.nd.array(np.expand_dims(np.asarray(self.label), 1), self.ctx)]

    @staticmethod
    def _worker(worker_id, data_queue, result_queue, stop_word, stop_flag,get_data_function,get_augment,augment_ratio,data_type,
                crop_or_pad,is_train,shapes,label_shapes):

        for item in iter(data_queue.get, stop_word):
            if stop_flag == 1:
                break
            img1,img2,label,index = get_data_function(item,data_type)
            if is_train :
                img1, img2, label = crop_or_pad(img1, img2, label, shapes, is_train, data_type)
                if uniform(0,1) < augment_ratio:
                    img1, img2, label = get_augment(img1, img2, label ,data_type)
                labels = []
                for j in xrange(len(label_shapes)):
                    labels.append(cv2.resize(label, (label_shapes[j][1], label_shapes[j][0])))
                result_queue.put((img1, img2, labels, index))
            else:
                # privide original label,but we must resize the image to match the input of the network.
                img1 = cv2.resize(img1,(shapes[2],shapes[1]))
                img2 = cv2.resize(img2,(shapes[2],shapes[1]))
                result_queue.put((img1, img2, label, index))

    def getindex(self):
        return self.current

    def reset(self):
        self.current = 0
        if self.be_shuffle:
            self.data_dirs = utils.shuffle(self.data_dirs)

        if self.multi_thread:
            self.shutdown()
            self._insert_queue()
            self._thread_start()

    def shutdown(self):
        # shutdown multi-process
        if self.multi_thread:
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

    def __init__(self, records, data_type, is_train, batch_shape, label_shapes, ctx, augment_ratio):

        super(multi_imageRecord, self).__init__()
        self.is_train = is_train
        self.data_type = data_type
        self.records = records
        self.augment_ratio = augment_ratio
        self.batch_size = batch_shape[0]
        self.shapes = batch_shape[1:]
        self.label_shapes = label_shapes
        self.ctx = ctx
        self.reset()

        self.provide_data = [('img1', batch_shape),('img2', batch_shape)]
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
        if self.is_train:
            self.labels = [[] for i in xrange(len(self.label_shapes))]
        else:
            self.labels = []

        if self.is_train:
            for j in xrange(self.batch_size):
                img1, img2, label = crop_or_pad(tmp1[j], tmp2[j], labels[j], self.shapes, True, self.data_type)
                img1 = (img1 * 0.0039216) - np.array([0.411451, 0.432060, 0.450141])
                img2 = (img2 * 0.0039216) - np.array([0.410602, 0.431021, 0.448553])
                self.img1.append(img1)
                self.img2.append(img2)
                if self.data_type == 'stereo':
                    label[label <= 0] = -100000
                for i in xrange(len(self.labels)):
                    self.labels[i].append(cv2.resize(label, (self.label_shapes[i][1], self.label_shapes[i][0])))
        else:
            for j in xrange(self.batch_size):
                img1 = cv2.resize(tmp1[j], (self.shapes[2], self.shapes[1]))
                img2 = cv2.resize(tmp2[j], (self.shapes[2], self.shapes[1]))
                self.img1.append(img1)
                self.img2.append(img2)
                self.labels.append(labels[j])

        return True

    def getdata(self):
        return [mx.nd.array(np.asarray(self.img1).transpose(0, 3, 1, 2), self.ctx),
                mx.nd.array(np.asarray(self.img2).transpose(0, 3, 1, 2), self.ctx)]

    def getlabel(self):
        #  output dimension of optical flow is 3,and stereo is 2
        if self.is_train:
            if len(self.labels[0][0].shape) == 3:
                return [mx.nd.array(np.asarray(self.labels[i]).transpose(0, 3, 1, 2), self.ctx) for i in xrange(len(self.labels))]
            else:
                return [mx.nd.array(np.expand_dims(np.asarray(self.labels[i]),3).transpose(0, 3, 1, 2), self.ctx)
                        for i in xrange(len(self.labels))]
        else:
            # when testing,it will only provide original labels
            if len(self.labels[0].shape) == 3:
                return [mx.nd.array(np.asarray(self.labels).transpose(0, 3, 1, 2), self.ctx)]
            else:
                return [mx.nd.array(np.expand_dims(np.asarray(self.labels), 3).transpose(0, 3, 1, 2), self.ctx)]

    def getindex(self):
        return self.records[0].getindex()

    def reset(self):
        for i in self.records:
            i.reset()


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

    rgb_cof1 = np.random.uniform(low=cfg.dataset.rgbmul[0], high=cfg.dataset.rgbmul[1], size=3)
    rgb_cof2 = np.random.uniform(low=cfg.dataset.rgbmul[0], high=cfg.dataset.rgbmul[1], size=3)
    gaussian_noise_scale = uniform(0.0, gaussian_noise)
    rotate = randint(-rotate_range, rotate_range)
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate, 1)

    tx = randint(int(-img1.shape[1] * translation_range), int(img1.shape[1] * translation_range))
    ty = randint(int(-img1.shape[0] * translation_range), int(img1.shape[0] * translation_range))
    M = np.float32([[1, 0, tx], [0, 1, ty]])

    beta = uniform(cfg.dataset.beta[0], cfg.dataset.beta[1])
    alpha = uniform(cfg.dataset.alpha[0], cfg.dataset.alpha[1])

    # multiply rgb factor plus guassian noise
    img1 = img1 * rgb_cof1 + np.random.normal(loc=0.0, scale=gaussian_noise_scale,size=img1.shape)
    img2 = img2 * rgb_cof2 + np.random.normal(loc=0.0, scale=gaussian_noise_scale,size=img1.shape)

    # rotation
    if data_type != 'stereo':
        img1 = cv2.warpAffine(img1, rotation_matrix, (cols, rows))
        img2 = cv2.warpAffine(img2, rotation_matrix, (cols, rows))
        label = cv2.warpAffine(label, rotation_matrix, (cols, rows))

    # translation
    img1 = cv2.warpAffine(img1, M, (cols, rows))
    img2 = cv2.warpAffine(img2, M, (cols, rows))
    label = cv2.warpAffine(label, M, (cols, rows))

    if data_type == 'stereo':
        label[label <= 0] = -100000

    # brightness and contrastness
    img1 = cv2.multiply(img1, np.array([alpha]))
    img1 = cv2.add(img1, np.array([beta]))
    img2 = cv2.multiply(img2, np.array([alpha]))
    img2 = cv2.add(img2, np.array([beta]))

    # scaling
    origin_shape = img1.shape[:2]
    if uniform(0, 1) < 0.5:
        y_shape = randint(1.0*origin_shape[0], cfg.dataset.scale[1]*origin_shape[0])
        x_shape = randint(1.0*origin_shape[1], cfg.dataset.scale[1]*origin_shape[1])
    else:
        y_shape = randint(cfg.dataset.scale[0] * origin_shape[0], 1.0 * origin_shape[0])
        x_shape = randint(cfg.dataset.scale[0] * origin_shape[1], 1.0 * origin_shape[1])
    img1 = cv2.resize(img1, (x_shape, y_shape))
    img2 = cv2.resize(img2, (x_shape, y_shape))
    if data_type == 'stereo':
        label = label * (x_shape/origin_shape[1])
    else:
        label[:, :, 0] = label[:, :, 0] * (x_shape/origin_shape[1])
        label[:, :, 1] = label[:, :, 1] * (y_shape/origin_shape[0])
    label = cv2.resize(label, (x_shape, y_shape))
    img1, img2, label = crop_or_pad(img1, img2, label, batchsize[1:], True, 'stereo')

    return img1, img2, label