from easydict import EasyDict as edict

cfg = edict()
cfg.MODEL = edict()
cfg.ADAM = edict()
cfg.dataset = edict()

cfg.MODEL.loss1_scale = 0.80
cfg.MODEL.loss2_scale = 0.10
cfg.MODEL.loss3_scale = 0.08
cfg.MODEL.loss4_scale = 0.05
cfg.MODEL.loss5_scale = 0.03
cfg.MODEL.loss6_scale = 0.01
cfg.MODEL.epoch_num = 100
cfg.MODEL.weight_init_scale = 0.03
cfg.MODEL.checkpoint_prefix = '/data/check_point/check_point/'

cfg.ADAM.beta1 = 0.9
cfg.ADAM.beta2 = 0.999
cfg.ADAM.epsilon = 1e-08

cfg.dataset.kitti_prefix = '/data/stereo&flow/kitti/'
cfg.dataset.flyingchairs_prefix = '/data/stereo&flow/FlyingChairs_release/data/'
cfg.dataset.SythesisData_prefix = '/rawdata/stereo/'

#augmentation
cfg.dataset.augment_ratio = 0.20
cfg.dataset.rotate_range = 10
cfg.dataset.translation_range = 0.10
cfg.dataset.gaussian_noise = 0.05

# beta: brightness    alpha :  contrastness
cfg.dataset.beta = (0,0.03)
cfg.dataset.alpha = (0.8,1.2)

# training batch size
batchsize  = (4,3,384,768)

