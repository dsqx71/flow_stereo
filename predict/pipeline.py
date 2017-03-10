import math
import os
import ConfigParser
import cv2
import mxnet as mx
import numpy as np
from PIL import Image

class Pipeline:
    """
    Pipeline for prediction

    Parameters
    ----------------
    config_path: str,
        directory of model config

    Examples
    ----------------
    # init pipeline
    piper = Pipeline(config_path = '/home/xudong/model_zoo/model.config')
    data = KittiDataset('stereo', '2015', is_train=False)

    # prediction
    for item in data.dirs:
        img1, img2, label, aux = data.get_data(item)
        dis = piper.process(img1, img2)
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
        self.original_shape = [config.getint('model','img_height'), config.getint('model', 'img_width')]

        self.width = int(math.ceil(self.original_shape[1] / 64.0) * 64)
        self.height = int(math.ceil(self.original_shape[0] / 64.0) * 64)

        if self.model_type not in ['stereo', 'flow']:
            raise ValueError('model prefix must be "stereo" or "flow"')

        self.ctx = mx.gpu(int(config.get('model', 'ctx')))
        model_path = os.path.join(base_folder, self.model_prefix)
        self.model = self.load_model(model_path)

    def load_model(self,model_path):

        net, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)
        self.arg_params = arg_params
        self.aux_params = aux_params
        new_arg_params = {}

        for k, v in arg_params.items():
            if k != 'img1' and k != 'img2' and not k.startswith('stereo'):
                new_arg_params[k] = v

        model = mx.model.FeedForward(ctx=self.ctx,
                                     symbol=net,
                                     arg_params=new_arg_params,
                                     aux_params=aux_params,
                                     numpy_batch_size=1)
        return model

    def preprocess_img(self, img1,img2):

        if isinstance(img1, Image.Image):
            img1 = np.asarray(img1)
        if isinstance(img2, Image.Image):
            img2 = np.asarray(img2)

        img1 = img1 * 0.00392156862745098
        img2 = img2 * 0.00392156862745098

        img1 = img1 - np.array([0.35372, 0.384273, 0.405834])
        img2 = img2 - np.array([0.353581, 0.384512, 0.406228])

        # cv2.resize doesn't support anti_alias
        img1 = cv2.resize(img1,(self.width, self.height))
        img2 = cv2.resize(img2,(self.width, self.height))

        img1 = np.expand_dims(img1.transpose(2, 0, 1), 0)
        img2 = np.expand_dims(img2.transpose(2, 0, 1), 0)

        return img1,img2

    def process(self, img1, img2):
        # The original_shape can be slightly different from self.original_shape
        original_height, original_width = img1.shape[:2]

        if self.need_preprocess:
            img1, img2 = self.preprocess_img(img1,img2)

        batch = mx.io.NDArrayIter(data = {'img1':img1,'img2':img2})
        pred = self.model.predict(batch)[0][0]

        if self.model_type == 'stereo':
            pred = pred * (original_width/float(self.width))

        elif self.model_type == 'flow':
            pred[0, :, :]  = pred[0, :, :] * (original_width / float(self.width))
            pred[1, :, :] =  pred[1, :, :] * (original_height / float(self.height))
            pred = pred.transpose(1, 2, 0)

        pred = cv2.resize(pred, (original_width, original_height))
        return pred