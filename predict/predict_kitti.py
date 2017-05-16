import argparse
from ..data import dataset, data_util
from .pipeline import Pipeline
# import flowstereo.pipe as pipe
from ..others import visualize
import numpy as np
import time

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy_config', type=str, help='directory of model config')
    # parser.add_argument('--output_prefix', type=str, help='prefix of output directory')
    parser.add_argument('--is_show', type=int, help='whether show results', default=False)
    args = parser.parse_args()

    # init pipeline
    piper = Pipeline(args.deploy_config)
    # data_type : 'stereo' or 'flow'
    model_type = piper.model_type
    # data = dataset.FlyingChairsDataset()
    data = dataset.KittiDataset(model_type, '2012', is_train=True)
    # data = dataset.SynthesisData(data_type=model_type,
    #                                  scene_list=['flyingthing3d'],
    #                                  rendering_level=['cleanpass'])
    # prediction
    error = 0.0
    count = 0.0
    for index, item in enumerate(data.dirs):

        img1, img2, label, aux = data.get_data(item)
        ret = piper.process(img1, img2)
        err = np.power(ret-label, 2)
        if model_type == 'flow':
            err = err.sum(axis=2)
        err = np.power(err, 0.5)
        error += err[err==err].sum()
        count += (err==err).sum()
        print 'index : ', index
        print 'EPE : ',(err[err==err].sum()/(err==err).sum())
        print 'total EPE : ', error/count

        if args.is_show:
            visualize.plot_pairs(img1, img2, ret, model_type, plot_patch=False)
        # data_util.writeKittiSubmission(ret, prefix=args.output_prefix, index=index, type=model_type)


