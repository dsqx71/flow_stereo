from easydict import EasyDict as edict
import mxnet as mx

cfg = edict()
cfg.MODEL = edict()
cfg.ADAM = edict()
cfg.dataset = edict()
cfg.RNN = edict()

#  model loss scale
cfg.MODEL.loss0_scale = 0.0
cfg.MODEL.loss1_scale = 0.01
cfg.MODEL.loss2_scale = 0.05
cfg.MODEL.loss3_scale = 0.10
cfg.MODEL.loss4_scale = 0.20
cfg.MODEL.loss5_scale = 0.40
cfg.MODEL.loss6_scale = 0.80
cfg.RNN.num_hidden = 128

cfg.MODEL.stereomatching_checkpoint = '/rawdata/check_point/baidu_stereomatching/stereo_matching'
cfg.MODEL.epoch_num = 5000
cfg.MODEL.weight_init_scale = 1.0
cfg.MODEL.checkpoint_prefix = '/rawdata/check_point/dispnet_corr/'

# otimizer setting
cfg.ADAM.beta1 = 0.90
cfg.ADAM.beta2 = 0.999
cfg.ADAM.epsilon = 1e-8
cfg.ADAM.weight_decay = 0.000

# training batch size
batchsize = (4, 3, 384, 768)
ctx = [mx.gpu(7),mx.gpu(6)]

#
cfg.dataset.gaussian_noise = 0.0005
cfg.dataset.rgbmul = (0.8, 1.2)
cfg.dataset.beta = (-0.7,0.2)
cfg.dataset.alpha =(0.8,1.2)
#  data prefix
cfg.dataset.kitti_prefix = '/rawdata/stereo/kitti/'
cfg.dataset.flyingchairs_prefix = '/rawdata/stereo/FlyingChairs_release/data/'#'/data01/FlyingChairs_release/data/'
cfg.dataset.SythesisData_prefix = '/rawdata/stereo/'#'/data01/'#
cfg.dataset.tusimple_stereo = '/data01/tusimple_stereo_data/'
cfg.dataset.prototxt_dir = '/home/xudong/dispflownet-release/models/DispNetCorr1D/model/data.prototxt'
cfg.dataset.prototxt_template = '/home/xudong/dispflownet-release/models/DispNetCorr1D/model/template.prototxt'
cfg.dataset.pretrain_caffe = '/home/xudong/dispflownet-release/models/DispNetCorr1D/model/DispNetCorr1D_CVPR2016.caffemodel'
cfg.dataset.solver_file = '/home/xudong/dispflownet-release/models/DispNetCorr1D/model/solver.prototxt'

cfg.dataset.mean_dir = '/rawdata/check_point/mean_{}.npy'
cfg.record_prefix = '/data01/stereo_rec/'
