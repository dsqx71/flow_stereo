"""
- After preprocessing, most patches in image1 should still seem similar to the corresponding patches in image2.
- current implementation of augmentation is very slow, it takes about 8ms to read and preprocessing data
"""
from ..data import dataset, data_util, augmentation
from ..others import visualize
from random import shuffle

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
    data_set = dataset.KittiDataset(data_type='flow', which_year='2015', is_train=True)

    augment_pipeline = augmentation.augmentation(max_num_tries=50,
                                                 cropped_height=320,
                                                 cropped_width=768,
                                                 data_type='flow',
                                                 augment_ratio=1.0,
                                                 noise_std=0.000001,
                                                 mirror_rate=0.5,
                                                 rotate_range={'method': 'uniform', 'low': -20, 'high': 20},
                                                 translate_range={'method': 'uniform', 'low': -0.2, 'high': 0.2},
                                                 zoom_range={'method': 'uniform', 'low': 0.8, 'high': 1.5},
                                                 squeeze_range={'method': 'uniform', 'low': 0.75, 'high': 1.25},
                                                 gamma_range={'method': 'uniform', 'low': 1.0, 'high': 1.0},
                                                 brightness_range={'method': 'normal', 'mean': 0.0, 'scale': 0.001},
                                                 contrast_range={'method': 'uniform', 'low': 1.0, 'high': 1.0},
                                                 rgb_multiply_range={'method': 'uniform', 'low': 1.0, 'high': 1.0},
                                                 interpolation_method='bilinear')
    shuffle(data_set.dirs)
    for item in data_set.dirs[:10]:
        img1, img2, label, aux = data_set.get_data(item)
        begin = time.time()
        img1, img2, label = augment_pipeline(img1, img2, label, discount_coeff = 1)
        print (time.time() - begin)
        visualize.plot_pairs(img1, img2, label, type='flow')


if __name__ == '__main__':

    test_crop()
    test_augmentation()
    # profile.run("test_augmentation()")
