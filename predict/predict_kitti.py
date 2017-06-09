import argparse
from ..data import dataset, data_util
from .pipeline import Pipeline
# import flowstereo.pipe as pipe
from ..others import visualize
import numpy as np
import time
import cv2

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy_config', type=str, help='directory of model config')
    parser.add_argument('--output_prefix', type=str, help='prefix of output directory')
    parser.add_argument('--is_show', type=int, help='whether show results', default=False)
    args = parser.parse_args()

    # config_path = '/rawdata/checkpoint_flowstereo/model_zoo/dispnetCSS_pretrain.config'
    # piper = Pipeline(config_path)
    #
    # # stereo: img1 is left image ,img2 is right image
    # # optical flow: img1 is the first frame,img2 is the second frame.
    #
    # img1 = cv2.imread('/home/xudong/000013_left.jpg')
    # img2 = cv2.imread('/home/xudong/000013_right.jpg')
    # img1 = cv2.resize(img1, (1824, 900))
    # img2 = cv2.resize(img2, (1824, 900))
    # # img1 = img1[(img1.shape[0]-768)/2:(img1.shape[0]-768)/2+768,100+ (img1.shape[1]-1024)/2:100+(img1.shape[1]-1024)/2+1024]
    # # img2 = img2[(img2.shape[0]-768)/2:(img2.shape[0]-768)/2+768,100+ (img2.shape[1]-1024)/2:100+(img2.shape[1]-1024)/2+1024]
    # ret = piper.process(img1,img2)
    #
    # # plot result
    # visualize.plot_pairs(img1, img2, ret, piper.model_type, plot_patch=False)

    # init pipeline
    piper = Pipeline(args.deploy_config)
    # data_type : 'stereo' or 'flow'
    model_type = piper.model_type
    # data = dataset.SynthesisData(data_type=model_type,
    #                              scene_list=['flyingthing3d'],
    #                              rendering_level=['cleanpass'],
    #                              is_train=False)
    # data = dataset.SintelDataSet(data_type='stereo', rendering_level='final', is_training=True)
    # data = dataset.FlyingChairsDataset()
    # data = dataset.KittiDataset(model_type, '2015', is_train=True)
    data = dataset.SynthesisData(data_type=model_type,
                                     scene_list=['flyingthing3d'],
                                     rendering_level=['cleanpass'])
    # prediction
    error = 0.0
    count = 0.0
    for index, item in enumerate(data.dirs):
        # if index == 104:
        #     continue
        img1, img2, label, aux = data.get_data(item)
        # img1 = img1[:,:,[2,1,0]]
        # img2 = img2[:,:,[2,1,0]]
        original_shape = label.shape
        # img1 = cv2.resize(img1, (2000, 700))
        # img2 = cv2.resize(img2, (2000, 700))
        ret = piper.process(img1, img2)
        # ret = cv2.resize(ret, original_shape[::-1]) * original_shape[1] / 2000.0
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

        if args.output_prefix:
            data_util.writeKittiSubmission(ret, prefix=args.output_prefix, index=index, type=model_type)


