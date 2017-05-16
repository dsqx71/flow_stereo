"""
- After preprocessing, most patches in image1 should still seem similar to the corresponding patches in image2.
- current implementation of augmentation is very slow, it takes about 8ms to read and preprocessing data
"""
from ..data import dataset, data_util, augmentation
from ..others import visualize
from random import shuffle

import numpy as np
import profile
import time

def test_crop():

    data_set = dataset.KittiDataset(data_type='stereo', which_year='2015', is_train=True)
    shuffle(data_set.dirs)
    assert len(data_set.dirs) == 200, 'wrong number'
    for item in data_set.dirs[:1]:
        img1, img2, label, aux = data_set.get_data(item)
        img1_cropped, img2_cropped, label_cropped = data_util.crop(img1, img2, label,
                                                                   target_height=320, target_width=768)
        assert img1_cropped.shape == img2_cropped.shape
        assert img1_cropped.shape[:2] == label_cropped.shape[:2]

        visualize.plot_pairs(img1_cropped, img2_cropped, label_cropped, type='stereo')

def test_augmentation():

    data_set = dataset.MultiDataset(data_type='flow')
    dataset_list = [dataset.KittiDataset('flow', '2015', is_train=True) for i in range(20)]
    # dataset_list.append(dataset.FlyingChairsDataset())
    #
    #                     # dataset.SynthesisData(data_type='flow',
    #                     #                       scene_list=['flyingthing3d'],
    #                     #                       rendering_level=['cleanpass']))
    data_set.register(dataset_list)
    augment_pipeline = augmentation.augmentation(
        interpolation_method='bilinear',
        max_num_tries=10,
        cropped_height=320,
        cropped_width=448,
        data_type='stereo',
        augment_ratio=0.8,
        mirror_rate=0.0,
        flip_rate = 0.0,
        noise_range = {'method':'uniform', 'exp':False, 'mean':0.03, 'spread':0.03},
        translate_range={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.4},
        rotate_range={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.4},
        zoom_range={'method': 'uniform', 'exp': True, 'mean': 0.2, 'spread': 0.4},
        squeeze_range={'method': 'uniform', 'exp': True, 'mean': 0, 'spread': 0.3},

        gamma_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},
        brightness_range={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.02},
        contrast_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},
        rgb_multiply_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},

        lmult_pow={'method': 'uniform', 'exp': True, 'mean': -0.2, 'spread': 0.4},
        lmult_mult={'method': 'uniform', 'exp': True, 'mean': 0.0, 'spread': 0.4},
        lmult_add={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.03},

        sat_pow={'method': 'uniform', 'exp': True, 'mean': 0, 'spread': 0.4},
        sat_mult={'method': 'uniform', 'exp': True, 'mean': -0.3, 'spread': 0.5},
        sat_add={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.03},

        col_pow={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.4},
        col_mult={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.2},
        col_add={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.02},

        ladd_pow={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.4},
        ladd_mult={'method': 'normal', 'exp': True, 'mean': 0.0, 'spread': 0.4},
        ladd_add={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.04},
        col_rotate={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 1})

    shuffle(data_set.dirs)
    for item in data_set.dirs[:10]:
        img1, img2, label, aux = data_set.get_data(item)
        begin = time.time()
        img1, img2, label = augment_pipeline(img1, img2, label, discount_coeff = 1)
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        print label.shape
        # print (time.time() - begin)
        # visualize.plot(img1,'img1')
        visualize.plot_pairs(img1, img2, label, type='flow')


if __name__ == '__main__':

    # test_crop()
    test_augmentation()
    # profile.run("test_augmentation()")
