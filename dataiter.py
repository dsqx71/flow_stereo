from collections import namedtuple
import cv2
import mxnet as mx
import numpy as np
from sklearn import utils
from config import cfg
from random import randint,uniform,gauss
import Queue
import multiprocessing as mp
import atexit
import logging

DataBatch = namedtuple('DataBatch', ['data', 'label', 'pad', 'index'])
class Dataiter(mx.io.DataIter):
    """
        data iterator of stereo and flow
    """
    def __init__(self, dataset, batch_size, is_train, label_shapes, augment_ratio, multi_thread=False,n_thread=10,be_shuffle=True,sub_mean=True):
        """
        Create a data iterator

        Parameters
        ----------
        dataset : dataset , please refer dataset.py
        batch_size : int
        is_train: bool
                If false,it will not provide label and augment data.
        label_shapes : dict
                flownet have 6 loss layer with different size,you can user util.estimate_label_size to estimate size
        augment_ratio : float
        multi_thread  : bool
        n_thread : int
        be_shuffle :bool
        sub_mean : bool
                indicate whether to subduce mean and divided by std
        """

        super(Dataiter, self).__init__()
        self.batch_size = batch_size[0]
        self.shapes = batch_size[1:]
        self.be_shuffle = be_shuffle

        if self.be_shuffle :
            self.data_dirs  = utils.shuffle(dataset.dirs)
        else:
            self.data_dirs  = dataset.dirs

        self.num_imgs  = len(self.data_dirs)
        self.is_train  = is_train
        self.label_shapes = label_shapes
        self.get_data_function = dataset.get_data
        self.current = 0
        self.data_type = dataset.data_type

        if self.is_train:
            self.augment_ratio = augment_ratio
        else:
            self.augment_ratio = 0

        self.sub_mean = sub_mean
        self.multi_thread = multi_thread
        self.stop_word = '==STOP--'
        self.n_thread = n_thread

        if self.multi_thread:
            self.worker_proc = None
            self.stop_flag = mp.Value('b', False)
            self.result_queue = mp.Queue(maxsize=self.batch_size*50)
            self.data_queue = mp.Queue()

    def _thread_start(self):

        self.stop_flag = False
        self.worker_proc = [mp.Process(target=Dataiter._worker,
                                       args=[pid,
                                             self.data_queue,
                                             self.result_queue,
                                             self.stop_word,
                                             self.stop_flag,
                                             self.get_data_function,
                                             Dataiter.get_augment,
                                             self.augment_ratio,
                                             self.sub_mean,
                                             self.data_type,
                                             Dataiter.crop_or_pad,
                                             self.is_train,
                                             self.shapes])
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

        if  self.current + self.batch_size > self.num_imgs:
            return False

        self.first_img = []
        self.second_img = []
        self.label = [[] for i in range(len(self.label_shapes))]

        for i in range(self.current,self.current+self.batch_size):
            if self.multi_thread:
                if self.is_train :
                    img1,img2,label,index = self.result_queue.get()
                else:
                    img1,img2,_,index = self.result_queue.get()
            else:
                if self.is_train :
                    img1,img2,label,index = self.get_data_function(self.data_dirs[i],self.sub_mean,self.data_type)
                    img1, img2, label = Dataiter.crop_or_pad(img1, img2, label, self.shapes, self.is_train,self.data_type)
                    if uniform(0,1) < self.augment_ratio:
                        img1, img2, label = Dataiter.get_augment(img1, img2, label,self.data_type )
                else:
                    img1,img2,_,index = self.get_data_function(self.data_dirs[i])
                    img1, img2, _ = Dataiter.crop_or_pad(img1, img2, None, self.shapes, self.is_train, self.data_type)

            self.first_img.append(img1)
            self.second_img.append(img2)
            if self.is_train :
                for j in range(len(self.label)):
                    self.label[j].append(cv2.resize(label, (self.label_shapes[j][1], self.label_shapes[j][0])))
        self.current += self.batch_size
        return True

    def getdata(self):
        return [np.asarray(self.first_img).swapaxes(3,2).swapaxes(2,1),
                np.asarray(self.second_img).swapaxes(3,2).swapaxes(2,1)]

    def getlabel(self):
        """
            Dispnet and Flownet have 6 loss layers with different size
        """
        if self.is_train:
            if len(self.label[0][0].shape) ==3:
                return [np.asarray(self.label[i]).swapaxes(3,2).swapaxes(2,1) for i in range(len(self.label))]
            else:
                return [np.expand_dims(np.asarray(self.label[i]),3).swapaxes(3,2).swapaxes(2,1) for i in range(len(self.label))]
        else :
            return None

    @staticmethod
    def _worker(worker_id, data_queue, result_queue, stop_word, stop_flag,get_data_function,get_augment,augment_ratio,sub_mean,data_type,
                crop_or_pad,is_train,shapes):
        count = 0
        for item in iter(data_queue.get, stop_word):
            if stop_flag == 1:
                break
            img1,img2,label,index = get_data_function(item,sub_mean,data_type)
            img1,img2,label = crop_or_pad(img1,img2,label,shapes,is_train,data_type)
            if uniform(0,1) < augment_ratio:
                img1, img2, label = get_augment(img1, img2, label ,data_type)
            result_queue.put((img1,img2,label,index))
            count += 1

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

    @staticmethod
    def get_augment(img1, img2, label,data_type):
        """
            rotation will destroy epipolar geometry ! so don't rotate stereo data
        """

        rows, cols, _ = img1.shape
        rotate_range = cfg.dataset.rotate_range
        translation_range = cfg.dataset.translation_range
        gaussian_noise = cfg.dataset.gaussian_noise

        gaussian_noise_scale = uniform(0.0, gaussian_noise)
        rotate = randint(-rotate_range, rotate_range)
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate, 1)

        tx = randint(int(-img1.shape[1] * translation_range), int(img1.shape[1] * translation_range))
        ty = randint(int(-img1.shape[0] * translation_range), int(img1.shape[0] * translation_range))
        M = np.float32([[1, 0, tx], [0, 1, ty]])

        beta = gauss(cfg.dataset.beta[0], cfg.dataset.beta[1])
        alpha = uniform(cfg.dataset.alpha[0], cfg.dataset.alpha[1])

        if data_type !='stereo':
            img1 = cv2.warpAffine(img1, rotation_matrix, (cols, rows))
            img2 = cv2.warpAffine(img2, rotation_matrix, (cols, rows))
            label = cv2.warpAffine(label, rotation_matrix, (cols, rows))

        img1 = cv2.warpAffine(img1, M, (cols, rows)) + np.random.normal(loc=0.0, scale=gaussian_noise_scale,
                                                                        size=img1.shape)
        img2 = cv2.warpAffine(img2, M, (cols, rows)) + np.random.normal(loc=0.0, scale=gaussian_noise_scale,
                                                                        size=img1.shape)
        label = cv2.warpAffine(label, M, (cols, rows))

        img1 = cv2.multiply(img1, np.array([alpha]))
        img1 = cv2.add(img1, np.array([beta]))

        img2 = cv2.multiply(img2, np.array([alpha]))
        img2 = cv2.add(img2, np.array([beta]))

        return img1, img2, label

    @staticmethod
    def crop_or_pad(img1,img2,label,shapes,is_train,data_type):

        y_ori,x_ori = img1.shape[:2]
        y,x = shapes[1:]

        if y>=y_ori and x >= x_ori:
            #padding
            tmp1 =  np.zeros((y, x, 3))
            tmp2 =  np.zeros((y, x, 3))

            if is_train:
                x_begin = randint(0, x-x_ori)
                y_begin = randint(0, y-y_ori)
            else:
                x_begin = 0
                y_begin = 0

            tmp1[y_begin : y_begin+y_ori, x_begin : x_begin + x_ori, :] = img1[:]
            tmp2[y_begin : y_begin+y_ori, x_begin : x_begin + x_ori, :] = img2[:]
            if label is not None:
                if data_type == 'stereo':
                    tmp3 = np.zeros((y, x))
                    tmp3[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori] = label[:]
                else:
                    tmp3 = np.zeros((y, x, 2))
                    tmp3[y_begin:y_begin + y_ori, x_begin:x_begin + x_ori,:] = label[:]

                return tmp1,tmp2,tmp3
            else:
                return tmp1,tmp2,None

        elif y<=y_ori and x <= x_ori:
            #cropping
            x_begin = randint(0, x_ori - x )
            y_begin = randint(0, y_ori - y )
            if label is not  None:
                return img1[y_begin:y_begin+y,x_begin:x_begin+x,:] ,img2[y_begin:y_begin+y,x_begin:x_begin+x,:],\
                    label[y_begin:y_begin+y,x_begin:x_begin+x]
            else:
                return img1[y_begin:y_begin+y,x_begin:x_begin+x,:] ,img2[y_begin:y_begin+y,x_begin:x_begin+x,:],None
        else:
            pass


