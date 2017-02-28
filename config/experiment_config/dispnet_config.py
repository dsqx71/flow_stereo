from easydict import EasyDict as edict
import mxnet as mx

cfg = edict()
cfg.model = edict()
cfg.loader = edict()
cfg.optimizer = edict()
cfg.dataset = edict()

# Basic setting
cfg.experiment_name='dispnet_adam_pretrain'
cfg.model.name = 'dispnet'
cfg.loader.name = 'caffeiter'
cfg.optimizer.name = 'adam'
cfg.dataset.name = 'kitti'


# Specific setting

# weight scale of loss in dispnet
cfg.model.loss1_scale = 0.01
cfg.model.loss2_scale = 0.05
cfg.model.loss3_scale = 0.10
cfg.model.loss4_scale = 0.20
cfg.model.loss5_scale = 0.40
cfg.model.loss6_scale = 0.80



dispnet_cfg.batchsize = (4, 3, 384, 768)