import math
import os
import ConfigParser
import cv2
import mxnet as mx
import numpy as np

class Pipeline:
    """
    Pipeline for prediction

    Parameters
    ----------------
    config_path: str,
        directory of deploy config

    Examples
    ----------------
    piper = Pipeline(config_path = '/home/xudong/model_zoo/model.config')
    data = KittiDataset('stereo', '2015', is_train=False)

    for item in data.dirs:
        img1, img2, label, aux = data.get_data(item)
        dis = piper.process(img1, img2)

    Notes:
        - inputs should be in BGR order
    """
    def __init__(self, config_path):

        config = ConfigParser.ConfigParser()
        config.read(config_path)
        # model base folder
        base_folder = os.path.split(os.path.abspath(config_path))[0]

        # prefix: 'stereo' or 'flow'
        self.model_type = config.get('model', 'model_type')
        self.model_prefix = config.get('model', 'model_prefix')
        self.need_preprocess = config.getboolean('model', 'need_preprocess')
        model_path = os.path.join(base_folder, self.model_prefix)

        tmp = np.load(model_path + '_mean.npy')
        self.mean1 = tmp[0:3]
        self.mean2 = tmp[3:6]

        if self.model_type not in ['stereo', 'flow']:
            raise ValueError('Allowable values of model prefix are "stereo" and "flow"')

        self.ctx = mx.gpu(int(config.get('model', 'ctx')))
        self.model = self.load_model(model_path)

        self.target_width = None
        self.target_height = None

    def load_model(self, model_path):

        net, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)
        self.arg_params = arg_params
        self.aux_params = aux_params
        new_arg_params = {}

        for k, v in arg_params.items():
            if k != 'img1' and k != 'img2' and 'label' not in k:
                new_arg_params[k] = v

        model = mx.model.FeedForward(ctx=self.ctx,
                                     symbol=net,
                                     arg_params=new_arg_params,
                                     aux_params=aux_params,
                                     numpy_batch_size=1)
        return model

    def preprocess_img(self, img1, img2):
        # image in BGR order
        img1 = img1 * 0.00392156862745098
        img2 = img2 * 0.00392156862745098

        img1 = img1 - self.mean1
        img2 = img2 - self.mean2

        img1 = cv2.resize(img1,(self.target_width, self.target_height))
        img2 = cv2.resize(img2,(self.target_width, self.target_height))

        img1 = np.expand_dims(img1.transpose(2, 0, 1), 0)
        img2 = np.expand_dims(img2.transpose(2, 0, 1), 0)

        return img1, img2

    def process(self, img1, img2):
        # target_shape >= original_shape
        original_height, original_width = img1.shape[:2]
        self.target_width = int(math.ceil(original_width / 64.0) * 64)
        self.target_height = int(math.ceil(original_height / 64.0) * 64)

        if self.need_preprocess:
            img1, img2 = self.preprocess_img(img1, img2)
        batch = mx.io.NDArrayIter(data = {'img1':img1, 'img2':img2})
        pred = self.model.predict(batch)
        if isinstance(pred, list):
            pred = pred[0]
        # postprocessing
        if self.model_type == 'stereo':
            pred = pred[0][0]
            pred = pred * (original_width/float(self.target_width))

        elif self.model_type == 'flow':
            # pred in the shape of  (1, 2, height, width)
            pred = pred[0]
            pred[0, :, :]  = pred[0, :, :] * (original_width / float(self.target_width))
            pred[1, :, :] =  pred[1, :, :] * (original_height / float(self.target_height))
            pred = pred.transpose(1, 2, 0)
        pred = cv2.resize(pred, (original_width, original_height))
        return pred

s1 = 6
s2 = 13

class PatchPipeline(object):
    """
    Pipeline for prediction
    Parameters
    ----------------
    config_path: str,
        directory of deploy config
    Examples
    ----------------
    Notes:
        - inputs should be in BGR order
    """
    def __init__(self, config_path):

        config = ConfigParser.ConfigParser()
        config.read(config_path)
        # model base folder
        base_folder = os.path.split(os.path.abspath(config_path))[0]

        # prefix: 'stereo' or 'flow'
        self.model_type = config.get('model', 'model_type')
        self.model_prefix = config.get('model', 'model_prefix')
        self.max_displacement = config.getint('model', 'max_displacement')
        self.batchsize = config.getint('model', 'batchsize')
        self.need_preprocess = config.getboolean('model', 'need_preprocess')
        model_path = os.path.join(base_folder, self.model_prefix)

        if self.model_type not in ['stereo', 'flow']:
            raise ValueError('Allowable values of model prefix are "stereo" and "flow"')

        self.ctx = mx.gpu(int(config.get('model', 'ctx')))
        self.model = self.load_model(model_path)

    def load_model(self, model_path):

        net, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)
        self.arg_params = arg_params
        self.aux_params = aux_params
        shape = (self.batchsize, 3, 13, 13)
        model = net.simple_bind(ctx=self.ctx,
                                grad_req='null',
                                left=shape,
                                right=shape,
                                left_downsample=shape,
                                right_downsample=shape)
        for key in model.arg_dict:
            if key in arg_params:
                model.arg_dict[key][:] = arg_params[key]
        return model

    def transform(self,patch):
        return patch.transpose(2,0,1)

    def preprocess_img(self, img1, img2):

        img1 = (img1 - img1.reshape(-1,3).mean(axis=0)) / img1.reshape(-1,3).std(axis=0)
        img2 = (img2 - img2.reshape(-1,3).mean(axis=0)) / img2.reshape(-1,3).std(axis=0)

        return img1, img2

    def process(self, img1, img2, points):

        if self.need_preprocess:
            img1, img2 = self.preprocess_img(img1, img2)

        valid_point = []
        data = []

        # Extract patches
        if self.model_type == 'stereo':
            for x, y in points:
                if x-s2>=0 and x+s2<=img1.shape[1] and y-s2>=0 and y+s2<=img1.shape[0]:
                    for u in range(0, self.max_displacement):
                        if x-s2-u>=0:
                            left_patch = self.transform(img1[y-s1:y+1+s1,x-s1:x+1+s1, :])
                            right_patch = self.transform(img2[y-s1:y+1+s1,x-s1-u:x+1+s1-u, :])
                            try:
                                left_downsample = self.transform(cv2.resize(img1[y-s2:y+s2,x-s2:x+s2,:], (0,0), fx=0.5, fy=0.5))
                                right_downsample = self.transform(cv2.resize(img2[y-s2:y+s2,x-s2-u:x+s2-u,:],(0,0), fx=0.5, fy=0.5))
                                data.append([left_patch, right_patch, left_downsample, right_downsample])
                                valid_point.append(((x, y), (x-u, y)))
                            except:
                                print y-s2,y+s2,x-s2,x+s2
                        else:
                            break

        # Dataiter
        data = np.array(data)
        data = mx.io.NDArrayIter(data={'left':data[:, 0],
                                       'right':data[:, 1],
                                       'left_downsample':data[:, 2],
                                       'right_downsample':data[:, 3]},
                                 batch_size=self.batchsize)

        # reorganize format
        pred = self.model.predict(data)[:,0]
        pred = np.split(pred, indices_or_sections=len(valid_point), axis=0)
        results = pred[:len(valid_point)]
        results = dict(zip(valid_point, results))
        ret = {}
        keys = list(results.keys())
        keys.sort()
        for key in keys:
            point = key[0]
            if point not in ret:
                ret[point] = []
            ret[point].append((key[1], np.asscalar(results[key])))

        # Sort similarities
        for key in ret:
            ret[key] = sorted(ret[key], cmp=lambda x,y: cmp(x[1], y[1]), reverse=True)
        return ret