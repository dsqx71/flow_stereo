from ..data import dataloader
from ..data import augmentation, dataset
from ..symbol import dispnet_symbol
from ..others import visualize
import time

def test_numpyloader():
    batchsize = (4, 3, 320, 768)
    loss_scale = {'loss1': 1.00, 'loss2': 0.20, 'loss3': 0.20, 'loss4':0.20, 'loss5':0.00,'loss6':0.00}
    net_symbol = dispnet_symbol.dispnet(loss_scale,'stereo', False)
    shapes = net_symbol.infer_shape(img1 = batchsize, img2= batchsize)
    label_shape =  zip(net_symbol.list_outputs(), shapes[1])
    augment_pipeline = augmentation.augmentation(max_num_tries=20,
                                                 cropped_height=160,
                                                 cropped_width=320,
                                                 data_type='stereo',
                                                 augment_ratio=1.0,
                                                 noise_std=0.01,
                                                 mirror_rate=0.0,
                                                 rotate_range={'method': 'uniform', 'low': -17, 'high': 17},
                                                 translate_range={'method': 'uniform', 'low': -0.2, 'high': 0.2},
                                                 zoom_range={'method': 'normal', 'mean': 1.0, 'scale': 0.5},
                                                 squeeze_range={'method': 'normal', 'mean': 1.0, 'scale': 0.2},
                                                 gamma_range={'method': 'normal', 'mean': 1.0, 'scale': 0.2},
                                                 brightness_range={'method': 'normal', 'mean': 0.0, 'scale': 0.04},
                                                 contrast_range={'method': 'normal', 'mean': 1.0, 'scale': 0.5},
                                                 rgb_multiply_range={'method': 'normal', 'mean': 1.0, 'scale': 0.5},
                                                 interpolation_method='bilinear')

    data_set = dataset.KittiDataset(data_type='stereo', which_year='2015', is_train=True)
    # label_shape = [('loss1_output', (8, 2, 384, 768))]

    dataiter = dataloader.numpyloader(experiment_name='2017_3_7',
                                      dataset=data_set,
                                      augmentation=augment_pipeline,
                                      batch_shape=batchsize,
                                      label_shape=label_shape,
                                      n_thread=15,
                                      half_life=200000,
                                      initial_coeff=0.5,
                                      final_coeff=1.0,
                                      interpolation_method='bilinear')

    dataiter.reset()
    tic = time.time()
    for batch in dataiter:
        print time.time() - tic
        img1 = batch.data[0].asnumpy()[0].transpose(1,2,0)
        img2 = batch.data[1].asnumpy()[0].transpose(1,2,0)
        label= batch.label[0].asnumpy()[0,0]
        # visualize.plot_pairs(img1, img2, label, 'stereo', plot_patch=False)
        tic = time.time()

if __name__ == '__main__':
    test_numpyloader()
