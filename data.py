from collections import namedtuple
import cv2
import mxnet as mx
import numpy as np
from sklearn import utils
from config import cfg
from random import randint,uniform,gauss

class KittiDataset:
    """
        stere and optical flow
    """
    def __init__(self, low, high, which='2012' , prefix=cfg.dataset.kitti_prefix):
        """
            low,high : index of a sample
        """
        self.dirs = []
        if which == '2015':
            gt_dir = 'disp_noc_0/'
            imgl_dir = 'image_2/'
            imgr_dir = 'image_3/'
        else:
            gt_dir = 'disp_noc/'
            imgl_dir = 'colored_0/'
            imgr_dir = 'colored_0/'

        for num in range(low, high):
            dir_name = '%06d' % num
            gt = prefix +  gt_dir + dir_name + '_10.png'.format(num)
            imgl = prefix + imgl_dir + dir_name + '_10.png'.format(num)
            imgr = prefix + imgr_dir + dir_name + '_10.png'.format(num)
            self.dirs.append((gt, imgl, imgr))

    @staticmethod
    def get_data(img_dir):
        """
            input : img_dir tuple :
            output : first_img , second_img , stereo label
        """
        dis  = np.round(cv2.imread(img_dir[0])/256.0).astype(int)
        left = cv2.imread(img_dir[1])
        right= cv2.imread(img_dir[2])

        left  = left * 0.0039216 - np.array([0.411451, 0.432060, 0.450141])
        right = right* 0.0039216 - np.array([0.410602, 0.431021, 0.448553])

        return left,right,dis


class FlyingChairsDataset:
    """
        dense optical flow dataset :
    """

    def __init__(self, low, high, prefix=cfg.dataset.flyingchairs_prefix):
        """
            low,high : index of a sample
        """
        self.dirs = []
        for i in range(low, high + 1):
            self.dirs.append([prefix + '%05d_img1.ppm' % i, prefix + '%05d_img2.ppm' % i, prefix + '%05d_flow.flo' % i])

    @staticmethod
    def shapes():
        return 3, 384, 512

    @staticmethod
    def get_augment(img1,img2,flow):

        rows,cols,_ = img1.shape
        rotate_range = cfg.dataset.rotate_range
        translation_range = cfg.dataset.translation_range
        gaussian_noise = cfg.dataset.gaussian_noise

        gaussian_noise_scale =  uniform(0.0,gaussian_noise)
        rotate = randint(-rotate_range,rotate_range)
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), rotate, 1)

        tx = randint(int(-img1.shape[1]*translation_range),int(img1.shape[1]*translation_range))
        ty = randint(int(-img1.shape[0]*translation_range),int(img1.shape[0]*translation_range))
        M = np.float32([[1,0,tx],[0,1,ty]])

        beta  = gauss(cfg.dataset.beta[0],cfg.dataset.beta[1])
        alpha = uniform(cfg.dataset.alpha[0],cfg.dataset.alpha[1])


        img1 = cv2.warpAffine(img1, rotation_matrix,(cols,rows))
        img2 = cv2.warpAffine(img2, rotation_matrix,(cols,rows))
        flow = cv2.warpAffine(flow, rotation_matrix,(cols,rows))

        img1 = cv2.warpAffine(img1,M,(cols,rows))  + np.random.normal(loc=0.0,scale=gaussian_noise_scale, size = img1.shape)
        img2 = cv2.warpAffine(img2,M,(cols,rows))  + np.random.normal(loc=0.0,scale=gaussian_noise_scale, size = img1.shape)
        flow = cv2.warpAffine(flow,M,(cols,rows))

        img1 = cv2.multiply(img1,np.array([alpha]))
        img1 = cv2.add(img1,np.array([beta]))

        img2 = cv2.multiply(img2,np.array([alpha]))
        img2 = cv2.add(img2,np.array([beta]))

        return img1,img2,flow

    @staticmethod
    def get_data(img_dir):
        """
            input : img_dir tuple :
            output : first_img , second_img , flow
        """

        tag_float = 202021.25

        first_img = cv2.imread(img_dir[0])
        second_img = cv2.imread(img_dir[1])

        with open(img_dir[2]) as f:

            nbands = 2
            tag = np.fromfile(f, np.float32, 1)[0]

            if tag != tag_float:
                raise ValueError('wrong tag possibly due to big-endian machine?')

            width = np.fromfile(f, np.int32, 1)[0]
            height = np.fromfile(f, np.int32, 1)[0]

            tmp = np.fromfile(f, np.float32)
            tmp = tmp.reshape(height, width*nbands)

            flow = np.zeros((height, width, 2))
            flow[:, :, 0] = tmp[:, 0::2]
            flow[:, :, 1] = tmp[:, 1::2]

            assert first_img.shape == second_img.shape
            assert first_img.shape[:2] == flow.shape[:2]


        first_img = first_img * 0.0039216 - np.array([0.411451, 0.432060, 0.450141])
        second_img = second_img * 0.0039216 - np.array([0.410602, 0.431021, 0.448553])

        return first_img, second_img, flow

DataBatch = namedtuple('DataBatch', ['data', 'label', 'pad', 'index'])
class FlowDataiter(mx.io.DataIter):

    def __init__(self, dataset, batch_size, data_type, label_shapes,augment_ratio):
        super(FlowDataiter, self).__init__()

        self.reset()
        self.batch_size = batch_size
        self.img_dirs = utils.shuffle(dataset.dirs)
        self.num_imgs = len(self.img_dirs)
        self.data_type = data_type
        self.label_shapes = label_shapes
        self.get_data_function = dataset.get_data

        self.first_img = []
        self.second_img = []
        self.flow = [[] for i in range(len(self.label_shapes))]

        self.inventory = 0
        self.pad = 0

        self.index = 0
        self.augment_ratio = augment_ratio
        self.get_augment = dataset.get_augment

    def iter_next(self):
        
        self.first_img = []
        self.second_img = []
        self.flow = [[] for i in range(len(self.label_shapes))]
        self.inventory = 0
        self.pad = 0
        
        for i in range(self.batch_size):
            
            if self.index < self.num_imgs:
                if self.data_type == 'test':
                    tmp1, tmp2 = self.get_data_function(self.img_dirs[self.index])
                    self.first_img.append(tmp1)
                    self.second_img.append(tmp2)
                    
                else:
                    tmp1, tmp2, tmp3 = self.get_data_function(self.img_dirs[self.index])
                    if uniform(0,1) < self.augment_ratio:
                        tmp1, tmp2, tmp3 = self.get_augment(tmp1, tmp2, tmp3 )
                    self.first_img.append(tmp1)
                    self.second_img.append(tmp2)
                    for j in range(len(self.flow)):
                        self.flow[j].append(cv2.resize(tmp3, (self.label_shapes[j][1], self.label_shapes[j][0])))
                self.inventory += 1
                self.index += 1
            else:
                self.pad += 1
                self.first_img.append(np.zeros_like(self.first_img[0]))
                self.second_img.append(np.zeros_like(self.second_img[0]))
                self.flow.append(np.zeros_like(self.flow[0]))
        
        if self.pad == 0:
            return True
        else:
            return False
            
    def getdata(self):
        
        return [np.asarray(self.first_img).swapaxes(3,2).swapaxes(2,1),
                np.asarray(self.second_img).swapaxes(3,2).swapaxes(2,1)]
    
    def getlabel(self):
        
        if self.data_type !='test':
            return [np.asarray(self.flow[i]).swapaxes(3,2).swapaxes(2,1) for i in range(len(self.flow))]
        else :
            return None
    
    def getindex(self):
        return self.index
    
    def getpad(self):
        return self.pad

    def reset(self):
        self.index = 0
