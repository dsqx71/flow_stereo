import glob

import os
from PIL import Image

import cv2
import numpy as np

from .config import cfg
from . import data_util

TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

class DataSet(object):
    """The base class of a dataset. The formats and organizations of datasets are often different from each other.
    The design purpose of this class is to decouple DataLoaders from specific datasets.

    Note:
      images are in BGR order

    It includes:
    - 'shapes': method, return shapes of dataset
    - 'get_data': method, read data, given by data directories
    - 'dirs': list of data directories
        the values of list can be list of str, like [img1_dir, img2_dir, lable_dir]
    - 'data_type': str, can be 'stereo' or 'flow'

    Examples
    ------------
    An example of directly using DataSet to read data:

        data_set = dataset.KittiDataset('stereo', '2015', is_train=True)
        for item in data_set.dirs:
            img1, img2, label, aux = data_set.get_data(item,'stereo')

    An example of collobarting with dataloader:

        data_set = dataset.KittiDataset('stereo', '2015', is_train=False)
        dataiter = dataloader.Dataiter_training(dataset=data_set,....)
    """

    def __init__(self):
        self.dirs = []
        self.data_type = None

    def get_data(self, img_dir):
        """
        Parameters
        ----------
        img_dir : list of str, [img1_dir, img2_dir, label_dir]
        data_type : str,
            can be 'stereo' or 'flow'

        return
        -------
        img1: array
            if data_type is 'stereo', img1 will be the left image, otherwise it will be the first frame
        img2: array
            if data_type is 'stereo', img2 will be the right image, otherwise it will be the second frame
        label: array
            if data_type is 'stereo', label will be disparity map, otherwise it will be optical flow field.
        aux: array
            auxiliary data
        """
        pass

    @property
    def shapes(self):
        """
        Returns
        ----------
        shapes: tuple,
            shapes of datasets
        """
        return None

class MultiDataset(DataSet):
    """
    MultiDataset allows users to combine several different kinds of dataset into a single one

    Parameters
    ----------
    data_type : str
        can be 'stereo' or 'flow', all datasets which added to MultiDataset should have the same data_type

    Examples
    ----------
        data_set = dataset.MultiDataset(data_type='flow')
        dataset_list = [dataset.KittiDataset('flow', '2015', is_train=True), dataset.FlyingChairsDataset()]
        data_set.register(dataset_list)
    """


    def __init__(self,data_type):

        super(MultiDataset, self).__init__()
        self.data_type=data_type
        self.dataset_registry = {}
        self.get_data_method = None

    def register(self, dataset_list):
        """
        Add datasets to MultiDataset

        Parameters
        ----------
        dataset_list : list of dataset
        """
        for dataset in dataset_list:
            assert self.data_type == dataset.data_type, 'inconsistant data_type'
            self.dataset_registry[dataset.__name__] = dataset
            for item in dataset.dirs:
                self.dirs.append([item, dataset.__name__])
            dataset.dirs = None

    def get_data(self, img_dir):

        dataset_type = img_dir[1]
        if dataset_type not in self.dataset_registry:
            raise ValueError("The dataset doesn't exist")
        self.get_data_method = self.dataset_registry[dataset_type].get_data
        img1, img2, label, aux = self.get_data_method(img_dir[0])

        return img1, img2, label, aux

class DataScheduler(DataSet):

    def __init__(self, data_type):

        super(DataScheduler, self).__init__()
        self.data_type = data_type
        self.dataset_registry = {}
        self.get_data_method = None

    def register(self, dataset_list, num_iteration):
        """
        Add datasets to DataScheduler

        Parameters
        ----------
        dataset_list : list of dataset
        iteration_list : list of int
        """
        for dataset in dataset_list:
            assert self.data_type == dataset.data_type, 'inconsistant data_type'
            self.dataset_registry[dataset.__name__] = {}
            self.dataset_registry[dataset.__name__]['dataset'] = dataset
            self.dataset_registry[dataset.__name__]['dirs'] = dataset.dirs
            dataset.dirs = None
        self.num_iteration = num_iteration

    def get_data(self, img_dir):

        dataset_type = img_dir[1]
        if dataset_type not in self.dataset_registry:
            raise ValueError("The dataset doesn't exist")
        self.get_data_method = self.dataset_registry[dataset_type].get_data
        img1, img2, label, aux = self.get_data_method(img_dir[0])

class SynthesisData(DataSet):
    """Synthesis dataset, it has three different scenes which have different baselines and focal length:
        - Driving: urban driving scence
        - flyingthing3d: randomly flying thing
        - Monkaa: 3D cartoon

    Two rendering levels:
        - clean pass
        - final pass(with blur)

    Parameters
    ----------
    data_type : str,
        can be 'stereo' or 'flow'
    scene_list : list of str,
        the allowable values of the list: 'Driving', 'flyingthing3d', 'Monkaa'
    rendering_level :  list of str
        the allowable values of the list: 'cleanpass', 'finalpass'
    prefix : str
        prefix of data directory

    Examples
    ------------
    data_set = dataset.SythesisData(data_type='stereo',
                                    scene_list =  ['flyingthing3d', 'Driving'],
                                    rendering_level=['cleanpass', 'finalpass'])

    References:
        [1]A large Dataset to train Convolutional networks for disparity, optical flow, and scene flow estimation
        [2]Offical description : http://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    """

    def __init__(self,  data_type, scene_list,
                 rendering_level,
                 prefix=cfg.dataset.SythesisData_prefix,
                 is_train=True):

        super(SynthesisData, self).__init__()
        self.data_type = data_type
        self.scene_list = scene_list
        self.rendering_level = rendering_level
        self.is_train = is_train
        self.get_dir(prefix)

        if self.is_train is False:
            assert len(self.scene_list) == 1 and self.scene_list[0] == 'flyingthing3d', 'Only flyingthing scene has testing set'

    def get_dir(self, prefix):

        # TODO : Synthesis Dataset has ground truth scene flow which is not supported here.
        if self.data_type == 'stereo':
            # Driving
            if 'Driving' in self.scene_list:
                for render_level in self.rendering_level:
                    for focallength in ('35', '15'):
                        for orient in ('forwards', 'backwards'):
                            for speed in ('fast', 'slow'):
                                img_dir = prefix + '{}/frames_{}/{}mm_focallength/scene_{}/{}/'.format('Driving',
                                                                                                       render_level,
                                                                                                       focallength,
                                                                                                       orient, speed)
                                label_dir = prefix + '{}/disparity/{}mm_focallength/scene_{}/{}/'.format('Driving',
                                                                                                         focallength,
                                                                                                         orient, speed)
                                num = len(glob.glob(img_dir + 'left/*'))
                                for i in range(1, num + 1):
                                    self.dirs.append([img_dir + 'left/%04d.png' % i,
                                                      img_dir + 'right/%04d.png' % i,
                                                      label_dir + 'left/%04d.pfm' % i])
            # Monkaa
            if 'Monkaa' in self.scene_list:
                for render_level in self.rendering_level:
                    scenes = glob.glob(prefix + '{}/frames_{}/*'.format('Monkaa', render_level))
                    for item in scenes:
                        scene = item.split('/')[-1]
                        num = len(glob.glob(prefix + '{}/frames_{}/{}/left/*'.format('Monkaa', render_level, scene)))
                        img_dir = prefix + '{}/frames_{}/{}/'.format('Monkaa', render_level, scene)
                        label_dir = prefix + '{}/disparity/{}/'.format('Monkaa', scene)
                        for i in range(0, num):
                            self.dirs.append([img_dir + 'left/%04d.png' % i,
                                              img_dir + 'right/%04d.png' % i,
                                              label_dir + 'left/%04d.pfm' % i])
            # flyingthing3d
            if 'flyingthing3d' in self.scene_list:
                for render_level in self.rendering_level:
                    style = 'TRAIN' if  self.is_train else 'TEST'
                    for c in ('A','B','C'):
                        num = glob.glob(prefix + '{}/frames_{}/{}/{}/*'.format('FlyingThings3D_release',
                                                                               render_level,
                                                                               style,
                                                                               c))
                        for item in num:
                            j = item.split('/')[-1]
                            img_dir = prefix + '{}/frames_{}/{}/{}/{}/'.format('FlyingThings3D_release',
                                                                               render_level,
                                                                               style,
                                                                               c,
                                                                               j)
                            label_dir = prefix + '{}/disparity/{}/{}/{}/'.format('FlyingThings3D_release',
                                                                                 style,
                                                                                 c,
                                                                                 j)
                            if 'TRAIN/C/0600/' in img_dir:
                                # TRAIN/C/0600/14 MISS
                                continue
                            for i in range(6, 16):
                                self.dirs.append([img_dir + 'left/%04d.png' % i,
                                                  img_dir + 'right/%04d.png' % i,
                                                  label_dir + 'left/%04d.pfm' % i])

        elif self.data_type == 'flow':
            if 'Driving' in self.scene_list:
                for render_level in self.rendering_level:
                    for focallength in ('35', '15'):
                        for orient in ('forwards', 'backwards'):
                            for speed in ('fast', 'slow'):
                                for h in ('left','right'):
                                    for time in ('into_future','into_past'):
                                        if speed == 'fast':
                                            num = 300
                                        else:
                                            num = 800
                                        img_dir = prefix + '{}/frames_{}/{}mm_focallength/scene_{}/{}/{}/'.format(
                                                                                                             'Driving',
                                                                                                             render_level,
                                                                                                             focallength,
                                                                                                             orient,
                                                                                                             speed,
                                                                                                             h)
                                        label_dir = prefix + '{}/optical_flow/{}mm_focallength/scene_{}/{}/{}/{}/'.format(
                                                                                                             'Driving',
                                                                                                             focallength,
                                                                                                             orient,
                                                                                                             speed,
                                                                                                             time, h)
                                        if h == 'left':
                                            sym = 'L'
                                        else:
                                            sym = 'R'

                                        if time == 'into_future':
                                            for i in range(1,num):
                                                self.dirs.append([img_dir + '%04d.png' % i,
                                                                  img_dir + '%04d.png' % (i+1),
                                                                  label_dir + 'OpticalFlowIntoFuture_%04d_%s.pfm' % (i, sym)])
                                        else:
                                            for i in range(1,num):
                                                self.dirs.append([img_dir + '%04d.png' % (i+1),
                                                                  img_dir + '%04d.png' % i,
                                                                  label_dir + 'OpticalFlowIntoPast_%04d_%s.pfm' % (i+1, sym)])

            #  flyingthings3D
            if 'flyingthing3d' in self.scene_list:
                for render_level in self.rendering_level:
                    style = 'TRAIN' if  self.is_train else 'TEST'
                    for c in ('A', 'B', 'C'):
                        num = glob.glob(prefix + '{}/frames_{}/{}/{}/*'.format('FlyingThings3D_release',
                                                                               render_level,
                                                                               style,
                                                                               c))
                        for item in num:
                            j = item.split('/')[-1]
                            for orient in ('left','right'):

                                if orient == 'left':
                                    sym = 'L'
                                else:
                                    sym = 'R'

                                for time in ('into_future', 'into_past'):

                                    img_dir = prefix +'{}/frames_{}/{}/{}/{}/{}/'.format('FlyingThings3D_release',render_level,
                                                                                 style,c,j,orient)
                                    label_dir =prefix + '{}/optical_flow/{}/{}/{}/{}/{}/'.format('FlyingThings3D_release',style,
                                                                                        c,j,time,orient)
                                    if 'TRAIN/C/0600/' in img_dir:
                                        continue

                                    for i in range(6,15):

                                        if time == 'into_future':
                                            self.dirs.append([img_dir + '%04d.png' % i,
                                                              img_dir + '%04d.png' % (i + 1),
                                                              label_dir + 'OpticalFlowIntoFuture_%04d_%s.pfm' % (i, sym)])
                                        else:
                                            self.dirs.append([img_dir + '%04d.png' % (i + 1),
                                                              img_dir + '%04d.png' % i,
                                                              label_dir + 'OpticalFlowIntoPast_%04d_%s.pfm' % (
                                                              i + 1, sym)])
            #  Monkaa
            if 'Monkaa' in self.scene_list:
                for render_level in self.rendering_level:

                    scenes = glob.glob(prefix + '{}/frames_{}/*'.format('Monkaa', render_level))
                    for item in scenes:
                        scene = item.split('/')[-1]

                        for orient in ('left','right'):

                            if orient == 'left':
                                sym = 'L'
                            else:
                                sym = 'R'

                            num = len(glob.glob(prefix + '{}/frames_{}/{}/left/*'.format('Monkaa', render_level, scene)))

                            img_dir = prefix + '{}/frames_{}/{}/{}/'.format('Monkaa', render_level, scene,orient)

                            for time in ('into_future', 'into_past'):
                                label_dir = prefix + '{}/optical_flow/{}/{}/{}/'.format('Monkaa',scene,time,orient)

                                for i in range(num-1):
                                    if time == 'into_future':
                                        self.dirs.append([img_dir + '%04d.png' % i,
                                                          img_dir + '%04d.png' % (i + 1),
                                                          label_dir + 'OpticalFlowIntoFuture_%04d_%s.pfm' % (i, sym)])
                                    else:
                                        self.dirs.append([img_dir + '%04d.png' % (i + 1),
                                                          img_dir + '%04d.png' % i,
                                                          label_dir + 'OpticalFlowIntoPast_%04d_%s.pfm' % (
                                                              i + 1, sym)])
    @property
    def shapes(self):
        return 540, 960

    @property
    def __name__(self):
        return SynthesisData.__name__

    def get_data(self, img_dir):

        img1 = cv2.imread(img_dir[0])
        img2 = cv2.imread(img_dir[1])

        if self.data_type == 'stereo':
            label = data_util.readPFM(img_dir[2])

        elif self.data_type == 'flow':
            label = data_util.readPFM(img_dir[2])
            label = label[:, :, :2]
        # synthesis dataset has no auxiliary data
        return img1, img2, label, None

class KittiDataset(DataSet):
    """
    KITTI Stereo/Flow DataSet, please refer to flow_stereo/docs/KITTI2015_description.txt for more information
    It has two parts:
        - KITTI 2012
        - KITTI 2015

    Parameters
    ----------
    data_type : str,
        can be 'stereo' or 'flow'
    which_year : str, default is '2015'
        can be '2015' or '2012'
    is_train : bool, default is True
        whether this is for training
    prefix : str
        prefix of dataset directory

    Examples
    ------------
        data_set = dataset.KittiDataset(data_type = 'stereo', which_year = '2015', is_train=True, prefix = ...)

    References:
        [1]Kitti stereo benchmark: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo
        [2]Development Kit: http://kitti.is.tue.mpg.de/kitti/devkit_scene_flow.zip
        [3]Andreas Geiger, Automatic Camera and Range Sensor Calibration using a single Shot

    Notes:
        - KITTI data have different shapes, probably because they were rectified by different calibration matrixes.
        - The ground truth of the dataset is sparse.
    """
    def __init__(self,data_type, which_year='2015', is_train=True, prefix=cfg.dataset.kitti_prefix):

        # TODO: KITTI 2015 contains ground truth scene flow which is not supported here.
        super(KittiDataset, self).__init__()
        self.dirs = []
        self.data_type = data_type
        self.is_train = is_train

        if which_year == '2012' and is_train==True:
            high = 194
        if which_year == '2012' and is_train==False:
            high = 195
        if which_year == '2015':
            high = 200

        if is_train == False:
            prefix = prefix + 'testing/'

        if self.data_type == 'stereo':
            if which_year == '2015':
                gt_dir = 'disp_occ_0/'
                imgl_dir = 'image_2/'
                imgr_dir = 'image_3/'
            else:
                gt_dir = 'disp_occ/'
                imgl_dir = 'colored_0/'
                imgr_dir = 'colored_1/'
            for num in range(0, high):
                dir_name = '%06d_10.png' % num
                gt = prefix +  gt_dir + '%06d_10.png' % num
                imgl = prefix + imgl_dir + dir_name
                imgr = prefix + imgr_dir + dir_name
                self.dirs.append([imgl, imgr, gt])
        else:
            if which_year == '2015':
                gt_dir = 'flow_occ_0/'
                img1_dir = 'image_2/'
                img2_dir = 'image_2/'
            else :
                gt_dir = 'flow_occ/'
                img1_dir = 'colored_0/'
                img2_dir = 'colored_0/'

            for num in range(0, high):
                dir_name = '%06d' % num
                gt = prefix + gt_dir + dir_name + '_10.png'.format(num)
                img1 = prefix + img1_dir + dir_name + '_10.png'.format(num)
                img2 = prefix + img2_dir + dir_name + '_11.png'.format(num)
                self.dirs.append([img1, img2, gt])
                
    @property
    def shapes(self):
        # The smallest shape in KITTI 2015
        return  370, 1224

    @property
    def __name__(self):
        return KittiDataset.__name__

    def get_data(self, img_dir):

        img1 = cv2.imread(img_dir[0])
        img2 = cv2.imread(img_dir[1])

        if self.is_train:
            if self.data_type == 'stereo':
                label = cv2.imread(img_dir[2],cv2.IMREAD_UNCHANGED).astype(np.float64)/256.0
                label[label <= 1] = np.nan
                return img1, img2, label, None
            elif self.data_type == 'flow' :
                bgr = cv2.imread(img_dir[2], cv2.IMREAD_UNCHANGED).astype(np.float64)
                # valid denotes if the pixel is valid or not (1 if true, 0 otherwise)
                valid = bgr[:, :, 0]
                label = bgr[:, :, 1:]
                label = (label - 32768.0) / 64.0
                tmp = np.zeros_like(label)
                tmp[:, :, 0] = label[:, :, 1]
                tmp[:, :, 1] = label[:, :, 0]
                tmp[valid==0, :] = np.nan
                label[:] = tmp
                return img1, img2, label, valid
        else:
            return img1, img2, None, None

class FlyingChairsDataset(DataSet):
    """
    This dataset only has optical flow data

    Parameters
    ----------
    prefix : str
        prefix of data directory

    References:
        [1] FlowNet: Learning Optical Flow with Convolutional Networks.
        [2] http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html
    """

    def __init__(self, prefix=cfg.dataset.flyingchairs_prefix):
        super(FlyingChairsDataset, self).__init__()
        self.data_type = 'flow'
        for i in range(1, 22873):
            self.dirs.append([prefix + '%05d_img1.ppm' % i, prefix + '%05d_img2.ppm' % i, prefix + '%05d_flow.flo' % i])

    @property
    def shapes(self):
        return 384, 512

    @property
    def __name__(self):
        return FlyingChairsDataset.__name__

    def get_data(self, img_dir):

        first_img = cv2.imread(img_dir[0])
        second_img = cv2.imread(img_dir[1])
        flow = data_util.readFLO(img_dir[2])

        return first_img, second_img, flow, None

class TusimpleDataset(DataSet):
    """
    The dataset only has ground truth disparity map

    Parameters
    ----------
    num_data : int
    prefix : str
        prefix of data directory

    Notes:
      - The ground truth disparity map in this dataset is very sparse, therefore you should consider using neareast
        neighbor instead of bilinear sampling, when interpolating.
    """
    def __init__(self,num_data,prefix=cfg.dataset.tusimple_stereo):

        super(TusimpleDataset, self).__init__()
        self.data_type = 'stereo'

        for i in range(0,num_data):
            self.dirs.append([prefix+'/left/%d.png' %i,
                              prefix+'/right/%d.png' %i,
                              prefix + '/disparity/%d.npy' % i])

    @property
    def shapes(self):
        return 680, 977

    @property
    def __name__(self):
        return TusimpleDataset.__name__

    def get_data(self, img_dir):

        img1 = cv2.imread(img_dir[0])[:680]
        img2 = cv2.imread(img_dir[1])[:680]

        label = np.load(img_dir[2])
        label = label.astype(np.float64)
        label[label<=0] = np.nan
        label = label[:680]

        return img1, img2, label, None



class SintelDataSet(DataSet):

    def __init__(self, data_type, rendering_level, is_training, prefix=cfg.dataset.sintel_prefix):

        self.data_type = data_type
        self.dirs = []
        if is_training is False and data_type == 'stereo':
            raise ValueError("Up till now, official stereo testing set hasn't been released!")

        mode = 'training' if is_training else 'testing'
        scene = glob.glob(os.path.join(prefix, mode, '{}_left/*'.format(rendering_level)))

        if data_type == 'stereo':
            for item in scene:
                img_list = glob.glob(item+'/*')
                img_list.sort()
                for img_path in img_list:
                    suffix = '/'.join(img_path.split('/')[-2:])
                    img1_dir = os.path.join(prefix, mode, '{}_left/'.format(rendering_level), suffix)
                    img2_dir = os.path.join(prefix, mode, '{}_right/'.format(rendering_level), suffix)
                    dis_dir = os.path.join(prefix, mode, 'disparities', suffix)
                    self.dirs.append([img1_dir, img2_dir, dis_dir])

    def get_data(self, img_dir):
        img1 = cv2.imread(img_dir[0])
        img2 = cv2.imread(img_dir[1])

        if self.data_type == 'stereo':
            label = SintelDataSet.disparity_read(img_dir[2])
        elif self.data_type == 'flow':
            label = SintelDataSet.flow_read(img_dir[3])

        return img1, img2, label, None

    @property
    def shapes(self):
        return 436, 1024

    @property
    def __name__(self):
        return SintelDataSet.__name__

    @staticmethod
    def flow_read(filename):
        """ Read optical flow from file, return (U,V) tuple.

        Original code by Deqing Sun, adapted from Daniel Scharstein.
        """
        f = open(filename,'rb')
        check = np.fromfile(f,dtype=np.float32,count=1)[0]
        assert check == TAG_FLOAT, ' flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
        width = np.fromfile(f,dtype=np.int32,count=1)[0]
        height = np.fromfile(f,dtype=np.int32,count=1)[0]
        size = width*height
        assert width > 0 and height > 0 and size > 1 and size < 100000000, ' flow_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
        tmp = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width*2))
        u = tmp[:,np.arange(width)*2]
        v = tmp[:,np.arange(width)*2 + 1]
        return u,v

    @staticmethod
    def disparity_read(filename):
        """ Return disparity read from filename. """
        f_in = np.array(Image.open(filename))
        d_r = f_in[:,:,0].astype('float64')
        d_g = f_in[:,:,1].astype('float64')
        d_b = f_in[:,:,2].astype('float64')

        depth = d_r * 4 + d_g / (2**6) + d_b / (2**14)
        return depth
