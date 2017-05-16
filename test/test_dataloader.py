from ..data import dataloader, LMDB
from ..data import augmentation, dataset
from ..symbol import dispnet_symbol
from ..others import visualize
import time
import mxnet as mx

def test_numpyloader():

    batchsize = (8, 3, 384, 768)
    loss_scale = {'loss1': 1.00, 'loss2': 0.20, 'loss3': 0.20, 'loss4':0.20, 'loss5':0.00,'loss6':0.00}
    net_symbol = dispnet_symbol.dispnet(loss_scale,'flow', False)
    shapes = net_symbol.infer_shape(img1 = batchsize, img2= batchsize)
    label_shape =  zip(net_symbol.list_outputs(), shapes[1])
    augment_pipeline = augmentation.augmentation(
        interpolation_method='bilinear',
        max_num_tries=10,
        cropped_height=384,
        cropped_width=768,
        data_type='flow',
        augment_ratio=1.0,
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

    data_set = dataset.SynthesisData(data_type='flow',
                          scene_list=['flyingthing3d'],
                          rendering_level=['cleanpass'])
    # label_shape = [('loss1_output', (8, 2, 384, 768))]

    dataiter = dataloader.numpyloader(ctx=[mx.gpu(3)],
                                      experiment_name='2017_3_7',
                                      dataset=data_set,
                                      augmentation=augment_pipeline,
                                      batch_shape=batchsize,
                                      label_shape=label_shape,
                                      n_thread=30,
                                      half_life=200000,
                                      initial_coeff=0.5,
                                      final_coeff=1.0,
                                      interpolation_method='bilinear')


    tic = time.time()
    for batch in dataiter:
        print time.time() - tic
        img1 = batch.data[0].asnumpy()[0].transpose(1,2,0)
        img2 = batch.data[1].asnumpy()[0].transpose(1,2,0)
        label= batch.label[0].asnumpy()[0,0]
        # visualize.plot_pairs(img1, img2, label, 'stereo', plot_patch=False)
        tic = time.time()


def test_lmdbloader():

    batchsize = (16, 3, 384, 512)
    loss_scale = {'loss1': 1.00, 'loss2': 0.20, 'loss3': 0.20, 'loss4':0.20, 'loss5':0.00,'loss6':0.00}
    net_symbol = dispnet_symbol.dispnet(loss_scale,'flow', False)
    shapes = net_symbol.infer_shape(img1 = batchsize, img2= batchsize)
    label_shape =  zip(net_symbol.list_outputs(), shapes[1])
    augment_pipeline = augmentation.augmentation(
        interpolation_method='bilinear',
        max_num_tries=10,
        cropped_height=384,
        cropped_width=512,
        data_type='stereo',
        augment_ratio=1.0,
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

    data_set = dataset.SynthesisData(data_type='stereo',
                                     scene_list=['flyingthing3d'],
                                     rendering_level=['cleanpass'])
    arguments = dict(#lmdb_path='/home/xudong/FlyingThings3D_release_TRAIN_lmdb/',
                    lmdb_path='/data/flyingthing_flow_lmdb/',
                    #lmdb_path= '/home/xudong/FlyingChairs_release_lmdb/',
                    data_type='flow',
                    # ctx=[mx.gpu(3)],
                    experiment_name='2017_5_9',
                    augmentation=augment_pipeline,
                    batch_shape=batchsize,
                    label_shape=label_shape,
                    interpolation_method='bilinear',
                    use_rnn=False,
                    rnn_hidden_shapes=None,
                    initial_coeff=0.1,
                    final_coeff=1.0,
                    half_life=50000,
                    chunk_size=4,
                    n_thread=15)

    dataiter = mx.io.PrefetchingIter([LMDB.lmdbloader(ctx=[mx.gpu(i)], **arguments) for i in range(7)])
    # dataiter = LMDB.lmdbloader(ctx=[mx.gpu(7)], **arguments)
    tic = time.time()
    for batch in dataiter:
        print '{} fps'.format(1.0 / (time.time() - tic))
        print len(batch.data)
        print batch.data[0].shape
        img1 = batch.data[0].asnumpy()[0].transpose(1,2,0)
        img2 = batch.data[1].asnumpy()[0].transpose(1,2,0)
        label= batch.label[0].asnumpy()[0].transpose(1,2,0)
        # visualize.plot_pairs(img1, img2, label, 'flow', plot_patch=True)
        tic = time.time()

if __name__ == '__main__':
    # test_numpyloader()
    test_lmdbloader()
