"""
The formats and organizations of datasets are different from each other,
therefore we define Dataset class which can decouple Dataloaders from specific datasets.
"""
import glob
import cv2
import numpy as np

from config.general_config import cfg
from utils import util

class DataSet:

    def __init__(self):
        """
            self.data_type : 'stereo' or 'flow'
            self.dirs : list of  file dirs
        """
        self.dirs = []
        self.data_type = None

    @staticmethod
    def shapes():
        """
            original shape
        """
        pass

    @staticmethod
    def get_data(img_dir, data_type):
        """
        Parameters
        ----------
        img_dir   : a tuple  (img1_dir, img2_dir, label_dir)
        data_type : 'stereo' or 'flow'

        return
        -------
        img1 , img2, label
        """
        pass


class SythesisData(DataSet):
    """
       CVPR 2016 :  A large Dataset to train Convolutional networks for disparity.optical flow,and scene flow estimation

       If you want to check setting, please refer :

            http://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    """

    def __init__(self,  data_type, data_list, prefix=cfg.dataset.SythesisData_prefix,which_version=['cleanpass']):

        self.dirs = []
        self.data_type = data_type
        self.data_list = data_list
        self.which_version = which_version
        self.get_dir(prefix)

    def get_dir(self, prefix):

        if self.data_type == 'stereo':

            # Driving
            if 'Driving' in self.data_list:
                for render_level in self.which_version:
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
                                for i in xrange(1, num + 1):
                                    self.dirs.append([img_dir + 'left/%04d.png' % i,
                                                      img_dir + 'right/%04d.png' % i,
                                                      label_dir + 'left/%04d.pfm' % i])
            # Monkaa
            if 'Monkaa' in self.data_list:
                for render_level in self.which_version:

                    scenes = glob.glob(prefix + '{}/frames_{}/*'.format('Monkaa', render_level))
                    for item in scenes:
                        scene = item.split('/')[-1]
                        num = len(glob.glob(prefix + '{}/frames_{}/{}/left/*'.format('Monkaa', render_level, scene)))
                        img_dir = prefix + '{}/frames_{}/{}/'.format('Monkaa', render_level, scene)
                        label_dir = prefix + '{}/disparity/{}/'.format('Monkaa', scene)
                        for i in xrange(0, num):
                            self.dirs.append([img_dir + 'left/%04d.png' % i,
                                              img_dir + 'right/%04d.png' % i,
                                              label_dir + 'left/%04d.pfm' % i])
            # flyingthing3d
            if 'flyingthing3d' in self.data_list:
                for render_level in self.which_version:
                    for style in ('TRAIN',):
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
                                for i in xrange(6, 16):
                                    self.dirs.append([img_dir + 'left/%04d.png' % i,
                                                      img_dir + 'right/%04d.png' % i,
                                                      label_dir + 'left/%04d.pfm' % i])

        else:
            if 'Driving' in self.data_list:
                for render_level in self.which_version:
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
                                            for i in xrange(1,num):
                                                self.dirs.append([img_dir + '%04d.png' % i,
                                                                  img_dir + '%04d.png' % (i+1),
                                                                  label_dir + 'OpticalFlowIntoFuture_%04d_%s.pfm' % (i, sym)])
                                        else:
                                            for i in xrange(1,num):
                                                self.dirs.append([img_dir + '%04d.png' % (i+1),
                                                                  img_dir + '%04d.png' % i,
                                                                  label_dir + 'OpticalFlowIntoPast_%04d_%s.pfm' % (i+1, sym)])

            #  flyingthings3D
            if 'flyingthing3d' in self.data_list:
                for render_level in self.which_version:
                    for style in ('TRAIN',):
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

                                        for i in xrange(6,15):

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
            if 'Monkaa' in self.data_list:
                for render_level in self.which_version:

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

                                for i in xrange(num-1):
                                    if time == 'into_future':
                                        self.dirs.append([img_dir + '%04d.png' % i,
                                                          img_dir + '%04d.png' % (i + 1),
                                                          label_dir + 'OpticalFlowIntoFuture_%04d_%s.pfm' % (i, sym)])
                                    else:
                                        self.dirs.append([img_dir + '%04d.png' % (i + 1),
                                                          img_dir + '%04d.png' % i,
                                                          label_dir + 'OpticalFlowIntoPast_%04d_%s.pfm' % (
                                                              i + 1, sym)])
    @staticmethod
    def shapes():
        return 540, 960

    def name(self):
        return 'synthesisData'

    @staticmethod
    def get_data(img_dir,data_type):

        img1 = cv2.imread(img_dir[0])
        img2 = cv2.imread(img_dir[1])

        if data_type == 'stereo':

            label, scale = util.readPFM(img_dir[2])
            label = label * scale

        elif data_type == 'flow':

            label, scale = util.readPFM(img_dir[2])
            label = label[:, :, :2]
            label = label * scale

        return img1, img2, label, img_dir[0].split('/')[-1]

class KittiDataset(DataSet):

    def __init__(self,data_type, which='2012', first_frame=True, is_train=True, prefix=cfg.dataset.kitti_prefix):

        if which == '2012' and first_frame == False:
            raise ValueError('kitti 2012 have not second frame ground truth')

        self.dirs = []
        self.data_type = data_type

        if which == '2012' and is_train:
            high = 194
        if which == '2012' and is_train==False:
            high = 195
        if which == '2015':
            high = 200

        if is_train == False:
            prefix = prefix + 'testing/'

        if self.data_type == 'stereo':
            if which == '2015':
                gt_dir = 'disp_occ_0/' if first_frame else 'disp_occ_1/'
                imgl_dir = 'image_2/'
                imgr_dir = 'image_3/'
            else:
                gt_dir = 'disp_occ/'
                imgl_dir = 'colored_0/'
                imgr_dir = 'colored_1/'
            for num in xrange(0, high):
                dir_name = '%06d_10.png' % num if first_frame else '%06d_11.png' % num 
                gt = prefix +  gt_dir + '%06d_10.png' % num
                imgl = prefix + imgl_dir + dir_name
                imgr = prefix + imgr_dir + dir_name
                self.dirs.append([gt, imgl, imgr])
        else:
            if which == '2015':
                gt_dir = 'flow_occ_0/'
                img1_dir = 'image_2/'
                img2_dir = 'image_2/'
            else :
                gt_dir = 'flow_occ/'
                img1_dir = 'colored_0/'
                img2_dir = 'colored_0/'

            for num in xrange(0, high):
                dir_name = '%06d' % num
                gt = prefix + gt_dir + dir_name + '_10.png'.format(num)
                img1 = prefix + img1_dir + dir_name + '_10.png'.format(num)
                img2 = prefix + img2_dir + dir_name + '_11.png'.format(num)
                self.dirs.append([gt, img1, img2])
                
    @staticmethod
    def shapes():
        # kitti data have different shapes, partly because they were rectified with different calibration matrixes.
        return 370,1224

    @staticmethod
    def name():
        return 'kitti'

    @staticmethod
    def get_data(img_dir, data_type):

        """
            input : img_dir tuple :
            output : first_img , second_img , stereo label
        """
        img1 = cv2.imread(img_dir[1])
        img2 = cv2.imread(img_dir[2])
        try:
            if data_type == 'stereo':
                label = cv2.imread(img_dir[0],cv2.IMREAD_UNCHANGED)/256.0
                label = label.astype(np.float64)
                label[label <= 3] = np.nan
                return img1, img2, label, None
            else :
                label = cv2.imread(img_dir[0],cv2.IMREAD_UNCHANGED)
                valid = label[:, :, 0]
                label = label[:, :, 1:]
                label = (label - 2 ** 15) / 64.0
                tmp = np.zeros_like(label)
                tmp[:, :, 0] = label[:, :, 1]
                tmp[:, :, 1] = label[:, :, 0]
                tmp[valid,:] = np.nan
                label = tmp
                return img1, img2, label, valid
            # train set or validation set have labels
        except :
            # test set don't have labels
            return img1, img2, None, None


class FlyingChairsDataset(DataSet):
    """
        CVPR 2015: FlowNet:Learing Optical Flow with Convolutional Networks

        This dataset only has optical flow !
    """

    def __init__(self, prefix=cfg.dataset.flyingchairs_prefix):
        """
            low,high : index of a sample
        """
        self.dirs = []
        for i in range(1, 22873):
            self.dirs.append([prefix + '%05d_img1.ppm' % i, prefix + '%05d_img2.ppm' % i, prefix + '%05d_flow.flo' % i])
        self.data_type = 'flow'

    @staticmethod
    def shapes():
        return 384, 512

    @staticmethod
    def name():
        return 'FlyingChairs'

    @staticmethod
    def get_data(img_dir, data_type):
        """
            input : img_dir tuple :
            output : first_img , second_img , flow
        """
        if data_type == 'stereo':
            raise ValueError("This dataset only has optical flow !")

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

        return first_img, second_img, flow, None


class TusimpleDataset(DataSet):

    def __init__(self,num_data,prefix=cfg.dataset.tusimple_stereo):

        self.data_type = 'stereo'
        self.dirs = []
        for i in range(0,num_data):
            self.dirs.append([prefix+'/disparity/%d.npy' % i,
                              prefix+'/left/%d.png' %i,
                              prefix+'/right/%d.png' %i])

    @staticmethod
    def shapes():
        return 680, 977

    @staticmethod
    def name():
        return 'TusimpleDataset'

    @staticmethod
    def get_data(img_dir, data_type):
        """
            input : img_dir tuple
            output : first_img , second_img , stereo label
        """
        img1 = cv2.imread(img_dir[1])[:680]
        img2 = cv2.imread(img_dir[2])[:680]

        label = np.load(img_dir[0])
        label = label.astype(np.float64)
        label[label<=0] = np.nan
        label = label[:680]

        assert img1.shape[:2] == img2.shape[:2]
        assert img1.shape[:2] == label.shape

        return img1, img2, label, None


class multidataset(DataSet):

    def __init__(self,data_type):

        self.data_type=data_type
        self.dirs = []

    def add_dataset(self, dataset):

        for item in dataset.dirs:
            self.dirs.append([item,dataset.name()])

    @staticmethod
    def shapes():
        return "unknown"

    @staticmethod
    def name():
        return 'multidataset'

    @staticmethod
    def get_data(img_dir, data_type):

        dataset_type = img_dir[1]

        if dataset_type=='kitti':
            get_data_method = KittiDataset.get_data

        elif dataset_type=='synthesisData':
            get_data_method = SythesisData.get_data

        elif dataset_type=='FlyingChairs':
            get_data_method = FlyingChairsDataset.get_data

        elif dataset_type=='TusimpleDataset':
            get_data_method = TusimpleDataset.get_data
        else:
            raise ValueError("the get data method don't exist")

        img1, img2, label, aux = get_data_method(img_dir[0],data_type)

        return img1, img2, label, aux