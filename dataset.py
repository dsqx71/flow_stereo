import cv2
import numpy as np
from config import cfg
import util
import glob
from skimage import io


class DataSet:

    def __init__(self):
        """
            self.data_type : 'stereo' or 'flow'
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
    def get_data(img_dir, sub_mean, data_type):
        """
        Parameters
        ----------
        img_dir   : a tuple  (img1_dir,img2_dir,label_dir)
        sub_mean  : indicate whether to subduce mean and divided by std
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

    def __init__(self,  data_type, prefix=cfg.dataset.SythesisData_prefix):

        self.dirs = []
        self.data_type = data_type
        self.get_dir(prefix)

    def get_dir(self, prefix):

        if self.data_type == 'stereo':

            # Driving
            for render_level in ('finalpass',):  # 'cleanpass'):
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
                                self.dirs.append((img_dir + 'left/%04d.png' % i,
                                                  img_dir + 'right/%04d.png' % i,
                                                  label_dir + 'left/%04d.pfm' % i))
            # Monkaa

            for render_level in ('finalpass',):  # 'cleanpass'):

                scenes = glob.glob(prefix + '{}/frames_{}/*'.format('Monkaa', render_level))
                for item in scenes:
                    scene = item.split('/')[-1]
                    num = len(glob.glob(prefix + '{}/frames_{}/{}/left/*'.format('Monkaa', render_level, scene)))
                    img_dir = prefix + '{}/frames_{}/{}/'.format('Monkaa', render_level, scene)
                    label_dir = prefix + '{}/disparity/{}/'.format('Monkaa', scene)
                    for i in range(0, num):
                        self.dirs.append((img_dir + 'left/%04d.png' % i,
                                          img_dir + 'right/%04d.png' % i,
                                          label_dir + 'left/%04d.pfm' % i))
            # flyingthing3d
            for render_level in ('finalpass',):  # 'cleanpass'):
                for style in ('TRAIN',):
                    for c in ('A', 'B', 'C'):
                        num = glob.glob(prefix + '{}/frames_{}/{}/{}/*'.format('FlyingThings3D',
                                                                               render_level,
                                                                               style,
                                                                               c))
                        for item in num:

                            j = item.split('/')[-1]
                            img_dir = prefix + '{}/frames_{}/{}/{}/{}/'.format('FlyingThings3D',
                                                                               render_level,
                                                                               style,
                                                                               c,
                                                                               j)
                            label_dir = prefix + '{}/disparity/{}/{}/{}/'.format('FlyingThings3D',
                                                                                 style,
                                                                                 c,
                                                                                 j)
                            for i in range(6, 16):
                                self.dirs.append((img_dir + 'left/%04d.png' % i,
                                                  img_dir + 'right/%04d.png' % i,
                                                  label_dir + 'left/%04d.pfm' % i))

        else:
            for render_level in ('finalpass',):  # 'cleanpass'):
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
                                            self.dirs.append((img_dir + '%04d.png' % i,
                                                              img_dir + '%04d.png' % (i+1),
                                                              label_dir + 'OpticalFlowIntoFuture_%04d_%s.pfm' % (i, sym)))
                                    else:
                                        for i in range(1,num):
                                            self.dirs.append((img_dir + '%04d.png' % (i+1),
                                                              img_dir + '%04d.png' % i,
                                                              label_dir + 'OpticalFlowIntoPast_%04d_%s.pfm' % (i+1, sym)))

            #  flyingthings3D
            '''
            for render_level in ('finalpass',):  # 'cleanpass'):
                for style in ('TRAIN',):
                    for c in ('A', 'B', 'C'):
                        num = glob.glob(prefix + '{}/frames_{}/{}/{}/*'.format('FlyingThings3D',
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

                                    img_dir = '{}/frames_{}/{}/{}/{}/{}/'.format('FlyingThings3D',render_level,
                                                                                 style,c,j,orient)
                                    label_dir ='{}/optical_flow/{}/{}/{}/{}/{}/'.format('FlyingThings3D',style,
                                                                                        c,j,time,orient)
                                    for i in range(6,15):
                                        if time == 'into_future':
                                            self.dirs.append((img_dir + '%04d.png' % i,
                                                              img_dir + '%04d.png' % (i + 1),
                                                              label_dir + 'OpticalFlowIntoFuture_%04d_%s.pfm' % (i, sym)))
                                        else:
                                            self.dirs.append((img_dir + '%04d.png' % (i + 1),
                                                              img_dir + '%04d.png' % i,
                                                              label_dir + 'OpticalFlowIntoPast_%04d_%s.pfm' % (
                                                              i + 1, sym)))
            '''
            #  Monkaa
            for render_level in ('finalpass',):  # 'cleanpass'):

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
                                    self.dirs.append((img_dir + '%04d.png' % i,
                                                      img_dir + '%04d.png' % (i + 1),
                                                      label_dir + 'OpticalFlowIntoFuture_%04d_%s.pfm' % (i, sym)))
                                else:
                                    self.dirs.append((img_dir + '%04d.png' % (i + 1),
                                                      img_dir + '%04d.png' % i,
                                                      label_dir + 'OpticalFlowIntoPast_%04d_%s.pfm' % (
                                                          i + 1, sym)))

    @staticmethod
    def shapes():
        return 3, 540, 960

    @staticmethod
    def get_data(img_dir, sub_mean, data_type):

        img1 = cv2.imread(img_dir[0])
        img2 = cv2.imread(img_dir[1])

        if sub_mean:
            img1 = (img1 - img1.reshape(-1, 3).mean(axis=0)) / img1.reshape(-1, 3).std(axis=0)
            img2 = (img2 - img2.reshape(-1, 3).mean(axis=0)) / img2.reshape(-1, 3).std(axis=0)
        if data_type == 'stereo':
            label, scale = util.readPFM(img_dir[2])
        else:
            label, scale = util.readPFM(img_dir[2])
            label = label[:, :, :2]

        return img1, img2, label, img_dir[0].split('/')[-1]


class KittiDataset(DataSet):
    """
        Kitti stereo and optical : 2015 and 2012
    """
    def __init__(self, low, high, data_type,which='2012', prefix=cfg.dataset.kitti_prefix):
        """
            low,high : index of a sample
        """
        self.dirs = []
        self.data_type = data_type

        if self.data_type == 'stereo':
            if which == '2015':

                gt_dir = 'disp_noc_0/'
                imgl_dir = 'image_2/'
                imgr_dir = 'image_3/'

            else:

                gt_dir = 'disp_noc/'
                imgl_dir = 'colored_0/'
                imgr_dir = 'colored_1/'

            for num in range(low, high):
                dir_name = '%06d' % num
                gt = prefix +  gt_dir + dir_name + '_10.png'.format(num)
                imgl = prefix + imgl_dir + dir_name + '_10.png'.format(num)
                imgr = prefix + imgr_dir + dir_name + '_10.png'.format(num)
                self.dirs.append((gt, imgl, imgr))
        else:

            if which == '2015':

                gt_dir = 'flow_noc_0/'
                img1_dir = 'image_2/'
                img2_dir = 'image_2/'

            else :

                gt_dir = 'flow_noc/'
                img1_dir = 'colored_0/'
                img2_dir = 'colored_0/'

            for num in range(low, high):
                dir_name = '%06d' % num
                gt = prefix + gt_dir + dir_name + '_10.png'.format(num)
                img1 = prefix + img1_dir + dir_name + '_10.png'.format(num)
                img2 = prefix + img2_dir + dir_name + '_11.png'.format(num)
                self.dirs.append((gt, img1, img2))

    @staticmethod
    def shapes():
        return 3, 375, 1242

    @staticmethod
    def get_data(img_dir, sub_mean, data_type):

        """
            input : img_dir tuple :
            output : first_img , second_img , stereo label
        """
        img1 = cv2.imread(img_dir[1])
        img2 = cv2.imread(img_dir[2])

        if sub_mean:
            img1 = (img1 - img1.reshape(-1, 3).mean(axis=0)) / img1.reshape(-1, 3).std(axis=0)
            img2 = (img2 - img2.reshape(-1, 3).mean(axis=0)) / img2.reshape(-1, 3).std(axis=0)

        if data_type == 'stereo':
            label = np.round(io.imread(img_dir[0])/256.0).astype(int)

        else :

            label = io.imread(img_dir[0])
            label = label.astype(np.float64)
            label[:, :, 0] = (label[:, :, 0] - 2 ** 15) / 64.0
            label[:, :, 1] = (label[:, :, 1] - 2 ** 15) / 64.0

        return img1, img2, label, img_dir[0].split('/')[-1]


class FlyingChairsDataset(DataSet):
    """
        CVPR 2015: FlowNet:Learing Optical Flow with Convolutional Networks

        This dataset only has optical flow !
    """

    def __init__(self, low, high, prefix=cfg.dataset.flyingchairs_prefix):
        """
            low,high : index of a sample
        """
        self.dirs = []
        for i in range(low, high + 1):
            self.dirs.append([prefix + '%05d_img1.ppm' % i, prefix + '%05d_img2.ppm' % i, prefix + '%05d_flow.flo' % i])
        self.data_type = 'flow'

    @staticmethod
    def shapes():
        return 3, 384, 512

    @staticmethod
    def get_data(img_dir, sub_mean, data_type):
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

        if sub_mean:
            first_img = (first_img - first_img.reshape(-1, 3).mean(axis=0)) / first_img.reshape(-1, 3).std(axis=0)
            second_img = (second_img - second_img.reshape(-1, 3).mean(axis=0)) / second_img.reshape(-1, 3).std(axis=0)

        return first_img, second_img, flow, img_dir[0].split('/')[-1]