import argparse
import mxnet as mx
from .pipeline import PatchPipeline
from ..data import dataset, data_util, augmentation, patchiter
from ..others import visualize
import numpy as np
import time

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy_config', type=str, help='directory of model config')
    parser.add_argument('--is_show', type=int, help='whether show results', default=False)
    args =  parser.parse_args()

    # Load model
    piper = PatchPipeline(args.deploy_config)
    model_type = piper.model_type
    data_set = dataset.KittiDataset(model_type, '2012', is_train=True)
    data_set.dirs = data_set.dirs[:1]
    dataiter = patchiter.patchiter(ctx=[mx.gpu(0)],
                                   experiment_name='test',
                                   dataset=data_set,
                                   img_augmentation=None,
                                   patch_augmentation=None,
                                   batch_size=2048,
                                   low=5,
                                   high=500,
                                   n_thread=20,
                                   be_shuffle=True)

    dataiter = mx.io.PrefetchingIter(dataiter)
    # time
    tot_time = 0.0
    count = 0
    for batch in dataiter:
        for index, item in enumerate(dataiter.provide_data):
            piper.model.arg_dict[item[0]] = batch.data[index]

        tic = time.time()
        piper.model.forward(is_train=False)
        tot_time = time.time() - tic
        count += 1
    print 'Avg time', tot_time / count / 2048 * 100 * 227







