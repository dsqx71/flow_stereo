import argparse
import logging
import mxnet as mx
import model
import dataiter
import util
import metric
import dataset
from config import cfg,batchsize,ctx

# parse parameter
parser = argparse.ArgumentParser()
parser.add_argument('--type',action='store',dest='type',type=str,choices=['stereo','flow'])
parser.add_argument('--continue', action='store', dest='con', type=int)
parser.add_argument('--lr', action='store', dest='lr', type=float)
cmd = parser.parse_args()

# logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

# load net and args
net = model.flow_and_stereo_net(cmd.type, is_sparse=False,is_l1=True)
label_shapes = util.estimate_label_size(net, batchsize)

if cmd.con == 0:
	#restart
	args = None ; auxs = None
else:
	#continue
	args, auxs = util.load_checkpoint(cfg.MODEL.checkpoint_prefix + cmd.type, cmd.con)
	logging.info("load the {} th epoch paramaters".format(cmd.con))

err = metric.EndPointErr()
if cmd.type == 'stereo':
	# image record
	records = dataset.get_imageRecord(dataset.SythesisData(cmd.type), batchsize[0], 25)
	data = dataiter.multi_imageRecord(records=records, data_type=cmd.type, is_train=True,
									  batch_size=batchsize, label_shapes=label_shapes, ctx=ctx[0],
									  augment_ratio=cfg.dataset.augment_ratio)
elif cmd.type == 'flow':
	#self-define iterator
	data = dataiter.Dataiter(dataset.SythesisData(data_type=cmd.type),
							 batch_size=batchsize,
							 is_train=True,
							 label_shapes=label_shapes,
							 augment_ratio=cfg.dataset.augment_ratio,
							 multi_thread=True,
							 n_thread=20,
							 be_shuffle=True,
							 sub_mean=True)

model = mx.model.FeedForward(ctx = ctx,
							 symbol=net,
							 num_epoch=cfg.MODEL.epoch_num,
							 optimizer = mx.optimizer.Adam(learning_rate=cmd.lr,beta1 = cfg.ADAM.beta1,beta2 = cfg.ADAM.beta2,
														   epsilon = cfg.ADAM.epsilon,wd = 0.00001,clip_gradient = 1000000),
							 initializer = mx.init.Normal(sigma=cfg.MODEL.weight_init_scale),
							 arg_params=args,
							 aux_params=auxs,
							 begin_epoch = cmd.con,
							 lr_scheduler = mx.lr_scheduler.FactorScheduler(15*50000/batchsize[0],0.5))

model.fit(X = data,
		  eval_metric=err,
		  epoch_end_callback=mx.callback.do_checkpoint(cfg.MODEL.checkpoint_prefix + cmd.type),
		  batch_end_callback=[mx.callback.Speedometer(batchsize[0],10)],
		  kvstore='local_allreduce_device')




