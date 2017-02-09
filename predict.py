import mxnet as mx
import ConfigParser
import os
from PIL import Image
import numpy as np
import cv2
import time
import math
from utils.util import outlier_sum
import dataset
import matplotlib.pyplot as plt

class Pipeline:

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
        self.width = int(math.floor(self.original_shape[1] * 1.0 / 64) * 64 )
        self.height = int(math.floor(self.original_shape[0] * 1.0 / 64) * 64 )

        if self.model_type not in ['stereo', 'flow']:
            raise ValueError('model prefix must be "stereo" or "flow"')

        self.ctx = mx.gpu(int(config.get('model', 'ctx')))
        model_path = os.path.join(base_folder, self.model_prefix)
        self.model = self.load_model(model_path)

    def load_model(self,model_path):

        net, arg_params, aux_params = mx.model.load_checkpoint(model_path,0)
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
        img1 = img1 - np.array([ 0.34641169 , 0.36324487,0.3527042 ])
        img2 = img2 - np.array([ 0.34641169 , 0.36324487,0.3527042 ])

        img1 = cv2.resize(img1,(self.width,self.height))
        img2 = cv2.resize(img2,(self.width,self.height))

        img1 = np.expand_dims(img1.transpose(2,0,1),0)
        img2 = np.expand_dims(img2.transpose(2,0,1),0)

        return img1,img2

    def process(self,img1,img2):

        original_height, original_width = img1.shape[:2]
        if self.need_preprocess:
            img1, img2 = self.preprocess_img(img1,img2)

        batch = mx.io.NDArrayIter(data = {'img1':img1,'img2':img2})
        pred = self.model.predict(batch)[0][0]

        if self.model_type == 'stereo':
            pred = pred * (original_width/float(self.width))

        elif self.model_type == 'flow':
            pred[0,:,:]  = pred[0,:,:] * (original_width/float(self.width))
            pred[1,:,:] =  pred[1,:,:] * (original_height/float(self.height))
            pred = pred.transpose(1,2,0)
        pred = cv2.resize(pred,(original_width,original_height))
        return pred

piper = Pipeline('/home/xudong/model_zoo/model.config')
data = dataset.KittiDataset('stereo', '2015', is_train=False)

tot_epe = 0
tot_d1a = 0
for index, item in enumerate(data.dirs):
    img1,img2,gt,_ = data.get_data(item,'stereo')
    dis = piper.process(img1,img2)
    # print dis.shape
    # mask = (gt == gt)
    # r = np.abs(gt-dis)
    # print r[mask].mean()
    # tot_epe += r[mask].mean()
    # d1all, outlier = outlier_sum(dis, gt, tau=3)
    # print d1all
    # tot_d1a += d1all

    dis = dis*256.0
    dis[dis<1.0] = 1.0
    dis = dis.astype(np.uint16)
    cv2.imwrite('/home/xudong/kitti2015_test/dispnet_drr/%06d_10.png' %  index, dis)
    # dis2 = cv2.imread('/home/xudong/kitti2015_test/dispnet_drr/%06d_10.png' %  index, cv2.IMREAD_UNCHANGED)
    # plt.figure()
    # plt.imshow(dis)
    # plt.waitforbuttonpress()
    # # print np.abs(dis-dis2).max()
    # plt.close()

print tot_epe/200.0 , tot_d1a/200.0