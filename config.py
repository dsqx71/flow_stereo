from easydict import EasyDict as edict
import mxnet as mx

cfg = edict()
cfg.MODEL = edict()
cfg.ADAM = edict()
cfg.dataset = edict()

cfg.MODEL.loss0_scale = 0.00
cfg.MODEL.loss1_scale = 0.000001
cfg.MODEL.loss2_scale = 0.0001
cfg.MODEL.loss3_scale = 0.01
cfg.MODEL.loss4_scale = 0.15
cfg.MODEL.loss5_scale = 0.20
cfg.MODEL.loss6_scale = 0.35

cfg.MODEL.epoch_num = 200
cfg.MODEL.weight_init_scale = 0.01
cfg.MODEL.checkpoint_prefix = '/rawdata/check_point/check_point/'

cfg.ADAM.beta1 = 0.9
cfg.ADAM.beta2 = 0.999
cfg.ADAM.epsilon = 1e-08

cfg.dataset.kitti_prefix = '/home/xudong/data/stereo&flow/kitti/'
cfg.dataset.flyingchairs_prefix = '/home/xudong/data/stereo&flow/FlyingChairs_release/data/'
cfg.dataset.SythesisData_prefix = '/rawdata/stereo/'
cfg.record_prefix = '/data01/stereo_rec/'

#augmentation
cfg.dataset.augment_ratio = 0.00
cfg.dataset.rotate_range = 5
cfg.dataset.translation_range = 0.00
cfg.dataset.gaussian_noise = 0.01

# beta: brightness    alpha :  contrastness
cfg.dataset.beta = (-0.01,0.01)
cfg.dataset.alpha = (0.80,1.20)
cfg.dataset.rgbmul = (0.8,1.2)

# training batch size
batchsize  = (8,3,384,768)

ctx  = [mx.gpu(i) for i in range(5,7)]
