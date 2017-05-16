"""
python -m flow_stereo.test.test_dataset

plot patches in image1 and corresponding patches in image2. The reasonable case is that most patches
in image1 should be similar to the counterpart in image2, except for occluded pixels.
"""
from ..data import dataset
from ..others import visualize
from random import shuffle
import time
import cv2

def test_SynthesisData():
    # check disparity
    # data_set = dataset.SynthesisData(data_type='stereo',
    #                                  scene_list=['flyingthing3d', 'Driving', 'Monkaa'],
    #                                  rendering_level=['cleanpass', 'finalpass'])
    # shuffle(data_set.dirs)
    # # assert len(data_set.dirs) > 70000, 'wrong number'
    #
    # for item in data_set.dirs[:10]:
    #     tic = time.time()
    #     img1, img2, label, aux = data_set.get_data(item)
    #     cv2.imwrite('/home/xudong/flowstereo-predictor/test/stereo/img1.png', img1)
    #     cv2.imwrite('/home/xudong/flowstereo-predictor/test/stereo/img2.png', img2)
    #     print time.time() - tic
    #     visualize.plot_pairs(img1, img2, label, 'stereo')


    # check optical flow field
    data_set = dataset.SynthesisData(data_type='flow',
                                     scene_list=['flyingthing3d'],
                                     rendering_level=['cleanpass'])
    shuffle(data_set.dirs)
    for item in data_set.dirs[:100]:
        tic = time.time()
        img1, img2, label, aux = data_set.get_data(item)
        print (time.time() - tic)
        # visualize.plot_pairs(img1, img2, label, 'flow')

def test_KittiData():
    # check disparity
    data_set = dataset.KittiDataset(data_type = 'stereo', which_year = '2015', is_train=True)
    shuffle(data_set.dirs)
    assert len(data_set.dirs) == 200, 'wrong number'
    for item in data_set.dirs[:1]:
        img1, img2, label, aux = data_set.get_data(item)
        visualize.plot_pairs(img1, img2, label, 'stereo')

    # check flow
    data_set = dataset.KittiDataset(data_type = 'flow', which_year = '2012', is_train=True)
    shuffle(data_set.dirs)
    for item in data_set.dirs[:1]:
        img1, img2, label, aux = data_set.get_data(item)
        visualize.plot_pairs(img1, img2, label, 'flow')

def test_FlyingChairData():
    # The dataset only has optical flow data
    data_set = dataset.FlyingChairsDataset()
    shuffle(data_set.dirs)
    assert len(data_set.dirs) ==  22872, 'wrong number'
    for item in data_set.dirs[:10]:
        tic = time.time()
        print item
        img1, img2, label, aux = data_set.get_data(item)
        cv2.imwrite('/home/xudong/flowstereo-predictor/test/flow/img1.png',img1)
        cv2.imwrite('/home/xudong/flowstereo-predictor/test/flow/img2.png',img2)
        print (time.time() - tic)
        visualize.plot_pairs(img1, img2, label, 'flow')

def test_TuSimpleData():
    # The dataset only has stereo data
    data_set = dataset.TusimpleDataset(4000)
    shuffle(data_set.dirs)
    for item in data_set.dirs[:1]:
        img1, img2, label, aux = data_set.get_data(item)
        visualize.plot_pairs(img1, img2, label, 'stereo')

def test_MultiDataSet():

    data_set = dataset.MultiDataset(data_type='flow')
    dataset_list = [dataset.KittiDataset('flow', '2015', is_train=True),
                    dataset.FlyingChairsDataset(),
                    dataset.SynthesisData(data_type='flow',
                                          scene_list=['flyingthing3d', 'Driving', 'Monkaa'],
                                          rendering_level=['cleanpass', 'finalpass'])
                    ]
    data_set.register(dataset_list)
    shuffle(data_set.dirs)
    for item in data_set.dirs[:1]:
        img1, img2, label, aux = data_set.get_data(item)
        visualize.plot_pairs(img1, img2, label, 'flow')


if __name__ == '__main__':
    # test_MultiDataSet()
    # test_TuSimpleData()
    test_SynthesisData()
    # test_FlyingChairData()
    # test_KittiData()
