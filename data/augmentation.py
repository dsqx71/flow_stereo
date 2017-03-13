import cv2
import random
import numpy as np

from . import data_util
from easydict import EasyDict as edict
from math import *

def between(data, low, high):
    """
    within the boundary or not
    """
    if low <= data and data <= high:
        return True
    else:
        return False

def clip(data, low, high):
    """
    Clip RGB data
    """
    data[data<low] = low
    data[data>high] = high
    return data

class augmentation(object):
    """
    Data augmentation. The design purpose of this class is to decouple augmentation from dataloaders

    Parameters
    ---------
     max_num_tries: int
        Not all spatial transformation coefficients are valid,
        max_num_tries denotes the maximum times it will try to find valid spatial coefficients.
     cropped_width: int
        target width
     cropped_height,
        target height
     data_type : str,
        the allowable values are 'stereo' and 'flow'
     augment_ratio :
        the probability of performing augmentation.
     noise_std: float,
        standard deviation of noise
     interpolation_method : str,
        how to interpolate data,
        the allowable values are 'bilinear' and 'nearest',
     mirror_rate: float
        the probability of performing mirror transformation
     rotate_range:
     translate_range:
     zoom_range:
     squeeze_range:
     gamma_range:
     brightness_range:
     contrast_range:
     rgb_multiply_range:
        dict, there are two forms of the dict:
        {'method':'uniform','low': float, 'high': float}, samples are uniformly distributed over the interval [low, high)
        or
        {'method':'normal', 'mean': float, 'scale': float}, samples are drawed from a normal distribution


    Notes:
    - For stereo matching, introducing any rotation, or vertical shift would break the epipolar constraint
    - When data type is 'stereo', it will ignore rotate_range, and not perform rotation transformation.

    Examples
    ----------
    augment_pipeline = augmentation.augmentation(max_num_tries=50,
                                                 cropped_height=320,
                                                 cropped_width=768,
                                                 data_type='stereo',
                                                 augment_ratio=1.0,
                                                 noise_std=0.0004,
                                                 mirror_rate=0.5,
                                                 rotate_range={'method': 'uniform', 'low': -17, 'high': 17},
                                                 translate_range={'method': 'uniform', 'low': -0.2, 'high': 0.2},
                                                 zoom_range={'method': 'uniform', 'low': 0.8, 'high': 1.5},
                                                 squeeze_range={'method': 'uniform', 'low': 0.75, 'high': 1.25},
                                                 gamma_range={'method': 'uniform', 'low': 0.9, 'high': 1.1},
                                                 brightness_range={'method': 'normal', 'mean': 0.0, 'scale': 0.002},
                                                 contrast_range={'method': 'uniform', 'low': 0.8, 'high': 1.2},
                                                 rgb_multiply_range={'method': 'uniform', 'low': 0.75, 'high': 1.25},
                                                 interpolation_method='bilinear')

    img1, img2, label = augment_pipeline(img1, img2, label, discount_coeff=0.5)
    """
    def __init__(self,
                 max_num_tries,
                 cropped_width,
                 cropped_height,
                 data_type,
                 augment_ratio,
                 noise_std,
                 interpolation_method,
                 mirror_rate,
                 rotate_range,
                 translate_range,
                 zoom_range,
                 squeeze_range,
                 gamma_range,
                 brightness_range,
                 contrast_range,
                 rgb_multiply_range):

        self.augment_ratio = augment_ratio
        self.data_type = data_type
        self.max_num_tries = max_num_tries
        self.cropped_width = cropped_width
        self.cropped_height = cropped_height

        if 'bilinear' in interpolation_method:
            self.interpolation_method = cv2.INTER_LINEAR
        elif 'nearest' in interpolation_method:
            self.interpolation_method = cv2.INTER_NEAREST
        else:
            raise ValueError("wrong interpolation method")

        # spatial transform
        self.mirror_rate = mirror_rate
        self.rotate_range = rotate_range
        self.translate_range = translate_range
        self.zoom_range = zoom_range
        self.squeeze_range = squeeze_range

        # chromatic transform
        self.gamma_range = gamma_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rgb_multiply_range = rgb_multiply_range
        self.noise_range = noise_std


    def generate_random(self, random_range, size=None):
        """
        random number generator
        """
        if random_range['method'] == 'uniform':
            discount = (random_range['high'] - random_range['low']) * (1.0 - self.discount_coeff) / 2.0
            low = random_range['low'] + discount
            high = random_range['high'] - discount
            result = np.random.uniform(low, high, size)
        elif random_range['method'] == 'normal':
            result = np.random.normal(loc=random_range['mean'],
                                      scale=random_range['scale']*self.discount_coeff,
                                      size=size)
        else:
            raise ValueError("wrong sampling method")
        return result

    def generate_spatial_coeffs(self):
        # order: mirror, rotate, translate, zoom, squeeze
        for i in range(self.max_num_tries):
            # identity matrix
            coeff = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])

            # mirror
            if random.uniform(0, 1) < self.mirror_rate * self.discount_coeff:
                mirror = np.array([[-1, 0,  0.5*self.cropped_width],
                                   [ 0, 1, -0.5*self.cropped_height],
                                   [ 0, 0, 1]])
            else:
                # move the center to (0, 0)
                mirror = np.array([[1, 0, -0.5*self.cropped_width],
                                   [0, 1, -0.5*self.cropped_height],
                                   [0, 0, 1]])
            coeff = np.dot(mirror, coeff)

            # rotate
            if self.data_type == 'flow':
                angle = self.generate_random(self.rotate_range) / 180.0 * pi
                rotate = np.array([[cos(angle), - sin(angle), 0],
                                   [sin(angle),   cos(angle), 0],
                                   [0, 0, 1]])
                coeff = np.dot(rotate, coeff)

            # translate
            dx = self.generate_random(self.translate_range)
            dy = self.generate_random(self.translate_range)
            translate = np.array([[1, 0, dx*self.cropped_width],
                                  [0, 1, dy*self.cropped_height],
                                  [0, 0, 1]])
            coeff = np.dot(translate, coeff)

            # zoom
            zoom_x = self.generate_random(self.zoom_range)
            zoom_y = zoom_x

            # squeeze
            squeeze_coeff = self.generate_random(self.squeeze_range)
            zoom_x *= squeeze_coeff
            zoom_y /= squeeze_coeff
            zoom = np.array([[1.0/zoom_x, 0, 0],
                             [0, 1.0/zoom_y, 0],
                             [0, 0, 1]])
            coeff = np.dot(zoom, coeff)

            # move_back
            move_back = np.array([[1, 0, self.width*0.5],
                                  [0, 1, self.height*0.5],
                                  [0, 0, 1]])
            coeff = np.dot(move_back, coeff)

            # Four corners should not exceed the boundaries of the origin
            flag = True
            for x in [0, self.cropped_width - 1]:
                for y in [0, self.cropped_height - 1]:
                    dest = np.array([x, y, 1])
                    src = np.dot(coeff, dest)
                    if between(src[0], 1, self.width - 2) == False or between(src[1], 1, self.height - 2) == False:
                        flag = False
                        break
            if flag:
                return coeff
        return None

    def spatial_transform(self, img1, img2, label):
        """
        coeff =  a1, a2, t1,
                 a3, a4, t2,
                 0,   0,  1
        a1, a2, a3, a4 : rotate, zoom, squeeze ; t1, t2 : crop and translate

        src_grid = np.dot(coeff, dst_grid)
        """
        # TODO: it does not support performing indepandent spaital transformation on the two images
        coeff = None
        for i in range(self.max_num_tries):
            coeff = self.generate_spatial_coeffs()
            if coeff is not None:
                break
        if coeff is not None:
            grid = np.zeros((3, self.cropped_height, self.cropped_width))
            xv, yv = np.meshgrid(np.arange(self.cropped_height), np.arange(self.cropped_width))
            grid[0, :, :] = yv.T
            grid[1, :, :] = xv.T
            grid[2, :, :] = 1.0
            grid = grid.reshape(3, -1)
            grid = np.dot(coeff, grid).astype(np.float32)
            grid = grid.reshape((3, self.cropped_height, self.cropped_width))
            img1_result = cv2.remap(img1, map1 = grid[0], map2=grid[1], interpolation=self.interpolation_method,
                                    borderValue=0)
            img2_result = cv2.remap(img2, map1=grid[0], map2=grid[1], interpolation=self.interpolation_method,
                                    borderValue=0)
            if label is not None :
                label_result = cv2.remap(label, map1=grid[0], map2=grid[1], interpolation=self.interpolation_method,
                                         borderValue=np.nan)
                if self.data_type == 'stereo':
                    label_result /= coeff[0,0]
                elif self.data_type == 'flow':
                    label_result = np.dot(label_result.reshape(-1,2), np.linalg.inv(coeff[:2, :2]).T)
                    label_result = label_result.reshape((self.cropped_height, self.cropped_width, 2))
            else:
                label_result = None

            return img1_result, img2_result, label_result
        else:
            print("Augmentation: Exceeded maximum tries in finding spatial coeffs.")
            img1_result, img2_result, label_result = data_util.crop(img1, img2, label, target_height=self.cropped_height,
                                                                                       target_width=self.cropped_width)
            return img1_result, img2_result, label_result

    def generate_chromatic_coeffs(self):

        coeff = edict()
        coeff.gamma = self.generate_random(self.gamma_range)
        coeff.brightness = self.generate_random(self.brightness_range)
        coeff.contrast = self.generate_random(self.contrast_range)
        coeff.rgb = np.array([self.generate_random(self.rgb_multiply_range) for i in range(3)])
        return coeff

    def apply_chromatic_transform(self, img, coeff):
        # normalize into [0, 1]
        img = img / 255.0
        # color change
        brightness_in = img.sum(axis=2)
        img = img * coeff.rgb
        brightness_out = img.sum(axis=2)
        brightness_coeff = brightness_in / (brightness_out + 1E-5)
        brightness_coeff = np.expand_dims(brightness_coeff, 2)
        brightness_coeff = np.concatenate([brightness_coeff for i in range(3)], axis=2)
        # compensate brightness
        img = img * brightness_coeff
        img = clip(img, 0, 1.0)
        # gamma change
        img = cv2.pow(img, coeff.gamma)
        # brightness change
        img = cv2.add(img, coeff.brightness)
        # contrast change
        img = 0.5 + (img-0.5) * coeff.contrast
        noise = np.zeros_like(img)
        cv2.randn(dst=noise, mean=0, stddev=self.noise_range)
        img += noise
        img = clip(img, 0.0, 1.0)
        img = img * 255
        return img

    def chromatic_transform(self, img1, img2):
        coeff = self.generate_chromatic_coeffs()
        img1 = self.apply_chromatic_transform(img1, coeff)
        img2 = self.apply_chromatic_transform(img2, coeff)
        return img1, img2

    def __call__(self, img1, img2, label, discount_coeff):
        """
        Perform data augmentation

        Parameters
        ----------
        img1: numpy.ndarray
        img2: numpy.ndarray
        label: numpy.ndarray
        discount_coeff : float
            the discount of augmentation coefficients
        """
        self.height = img1.shape[0]
        self.width = img1.shape[1]
        self.discount_coeff = discount_coeff

        if random.uniform(0, 1) < self.augment_ratio:
            img1, img2, label = self.spatial_transform(img1, img2, label)
            img1, img2 = self.chromatic_transform(img1, img2)

        return img1, img2, label

    # TODO: Eigenspace tranformation
    # TODO: render rain / frog / sun light
    # TODO: perform spatial transform on two images individually




