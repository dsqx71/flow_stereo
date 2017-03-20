from easydict import EasyDict as edict

cfg = edict()
cfg.dataset = edict()
cfg.model = edict()
cfg.imagerecord = edict()
cfg.caffe = edict()

#------------compulsory params-----------------
#You must download datasets and correctly set the directory prefix.
cfg.dataset.kitti_prefix = '/rawdata/stereo/kitti/'
cfg.dataset.flyingchairs_prefix = '/rawdata/stereo/FlyingChairs_release/data/'#'/home/xudong/flow_stereo/data/flyingchair/'
cfg.dataset.SythesisData_prefix = '/data01/'#'/rawdata/stereo/'#'/data01/'#'/home/xudong/flow_stereo/data/synthesis/'
cfg.dataset.tusimple_stereo = '/data01/tusimple_stereo_data/'

# mean file
cfg.dataset.mean_dir = '/rawdata/checkpoint_flowstereo/mean/'
# directory prefix of results
cfg.model.check_point = '/rawdata/checkpoint_flowstereo/'
# final results
cfg.model.model_zoo = "/rawdata/checkpoint_flowstereo/model_zoo/"
# log
cfg.model.log_prefix = '/rawdata/checkpoint_flowstereo/logs/'

#------------Optional params-------------------
# pretrain model
cfg.model.pretrain_model_prefix = '/rawdata/checkpoint_flowstereo/pretrain_model/'

# directories of caffe stuff. If you don't use caffeloader, ignore them
cfg.caffe.prototxt_dir = '/home/xudong/dispflownet-release/models/DispNetCorr1D/model/data.prototxt'
cfg.caffe.prototxt_template = '/home/xudong/dispflownet-release/models/DispNetCorr1D/model/template.prototxt'
cfg.caffe.pretrain_caffe = '/home/xudong/dispflownet-release/models/DispNetCorr1D/model/DispNetCorr1D_CVPR2016.caffemodel'
cfg.caffe.solver_file = '/home/xudong/dispflownet-release/models/DispNetCorr1D/model/solver.prototxt'


