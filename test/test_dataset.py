"""
python -m flow_stereo.test.test_dataset

plot patches in img1 and corresponding patches in img2. The reasonable case is that most patches
in img1 should be similar to the counterpart in img2, except for occluded pixels.
"""
from ..data import dataset
from ..utils import visualize
from random import shuffle

def plot(img1, img2, label, type):

    visualize.plot(img1, 'img1')
    visualize.plot(img2, 'img2')
    if type == 'stereo':
        visualize.plot(label, 'disparity')
    elif type == 'flow':
        color = visualize.flow2color(label)
        vector = visualize.flow2vector(label)
        visualize.plot(color, 'color')
        visualize.plot(vector, 'vector')
    visualize.check_data(img1, img2, type, label, interval=20, number=2, y_begin=100, x_begin=100)

def test_SynthesisData():
    # check disparity
    data_set = dataset.SynthesisData(data_type='stereo',
                                     scene_list=['flyingthing3d', 'Driving', 'Monkaa'],
                                     rendering_level=['cleanpass', 'finalpass'])
    shuffle(data_set.dirs)
    assert len(data_set.dirs) > 70000, 'wrong number'

    for item in data_set.dirs[:3]:
        img1, img2, label, aux = data_set.get_data(item)
        plot(img1, img2, label, 'stereo')

    # check optical flow field
    data_set = dataset.SynthesisData(data_type='flow',
                                     scene_list=['flyingthing3d', 'Driving', 'Monkaa'],
                                     rendering_level=['cleanpass', 'finalpass'])
    shuffle(data_set.dirs)
    for item in data_set.dirs[:3]:
        img1, img2, label, aux = data_set.get_data(item)
        plot(img1, img2, label, 'flow')

def test_KittiData():
    # check disparity
    data_set = dataset.KittiDataset(data_type = 'stereo', which_year = '2015', is_train=True)
    shuffle(data_set.dirs)
    assert len(data_set.dirs) == 200, 'wrong number'
    for item in data_set.dirs[:3]:
        img1, img2, label, aux = data_set.get_data(item)
        plot(img1, img2, label, 'stereo')

    # check flow
    data_set = dataset.KittiDataset(data_type = 'flow', which_year = '2012', is_train=True)
    shuffle(data_set.dirs)
    for item in data_set.dirs[:3]:
        img1, img2, label, aux = data_set.get_data(item)
        plot(img1, img2, label, 'flow')

def test_FlyingChairData():
    # The dataset only has optical flow data
    data_set = dataset.FlyingChairsDataset()
    shuffle(data_set.dirs)
    assert len(data_set.dirs) ==  22872, 'wrong number'
    for item in data_set.dirs[:3]:
        img1, img2, label, aux = data_set.get_data(item)
        plot(img1, img2, label, 'flow')

def test_TuSimpleData():
    # The dataset only has stereo data
    data_set = dataset.TusimpleDataset(4000)
    shuffle(data_set.dirs)
    for item in data_set.dirs[:3]:
        img1, img2, label, aux = data_set.get_data(item)
        plot(img1, img2, label, 'stereo')

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
    for item in data_set.dirs[:3]:
        img1, img2, label, aux = data_set.get_data(item)
        plot(img1, img2, label, 'flow')

def main():
    print ('''Please make sure that most patches in img1 should be similar to the counterpart in img2,
    except for occluded pixels.''')
    test_MultiDataSet()
    test_TuSimpleData()
    test_FlyingChairData()
    test_KittiData()
    test_SynthesisData()

if __name__ == '__main__':
    main()
