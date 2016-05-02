import argparse
import logging
import mxnet as mx
import model
import dataiter
import util
import metric
import dataset
from config import cfg,batchsize


# parse parameter

parser = argparse.ArgumentParser()
parser.add_argument('--type',action='store',dest='type',type=str,choices=['stereo','flow'])
parser.add_argument('--continue', action='store', dest='con', type=int)
parser.add_argument('--lr', action='store', dest='lr', type=float)
parser.add_argument('--ctx', action='store', dest='ctx', type=int,choices=[0,1,2,3])
cmd = parser.parse_args()

# logging

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
					datefmt='%a, %d %b %Y %H:%M:%S',filename='log_training_{}.log'.format(cmd.type), filemode='a')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# config

epoch = cfg.MODEL.epoch_num
ctx = mx.gpu(cmd.ctx)

# load net and args

net = model.flow_and_stereo_net(cmd.type, loss1_scale=cfg.MODEL.loss1_scale, loss2_scale=cfg.MODEL.loss2_scale,
									      loss3_scale=cfg.MODEL.loss3_scale, loss4_scale=cfg.MODEL.loss4_scale,
									      loss5_scale=cfg.MODEL.loss5_scale, loss6_scale=cfg.MODEL.loss6_scale)

if cmd.con == 0:
	#restart
	executor = net.simple_bind(ctx=ctx,grad_req='write',img1=batchsize ,img2=batchsize )
	util.init_param(cfg.MODEL.weight_init_scale , executor.arg_dict)
	logging.info("complete network architecture design")
else:
	#continue
	executor = util.load_model(name= cfg.MODEL.checkpoint_prefix + cmd.type,
							   epoch = cmd.con,
							   net=net,
							   batch_shape = batchsize ,
							   ctx = ctx,
							   network_type='write')
	logging.info("load the {} th epoch paramaters".format(cmd.con))


#init
keys = net.list_arguments()
grads = dict(zip(keys,executor.grad_arrays))
opt = mx.optimizer.Adam(learning_rate=cmd.lr,
						beta1=cfg.ADAM.beta1,
						beta2=cfg.ADAM.beta2,
						epsilon=cfg.ADAM.epsilon,
						rescale_grad=1.0/batchsize[0])

endpointerr = metric.EndPointErr()
label_shapes = util.estimate_label_size(net, batchsize )

#You can choose dataset here
iterator = dataiter.Dataiter(dataset= dataset.SythesisData(data_type= cmd.type),
                             batch_size = batchsize,
							 is_train = True,
							 label_shapes = label_shapes,
							 augment_ratio = cfg.dataset.augment_ratio,
							 multi_thread=True, n_thread=10, be_shuffle=True, sub_mean=True)
states = {}
for index, key in enumerate(executor.arg_dict):
	if 'img' not in key and 'stereo' not in key and 'flow' not in key:
		states[key] = opt.create_state(index, executor.arg_dict[key])

#train
for i in range(epoch):

	iterator.reset()
	endpointerr.reset()
	tot_err = 0
	count = 0
	iteration = 0

	for dbatch in iterator:

		iteration += 1
		executor.arg_dict['img1'][:] = dbatch.data[0]
		executor.arg_dict['img2'][:] = dbatch.data[1]
		for j in range(len(label_shapes)):
			executor.arg_dict['{}_downsample{}'.format(cmd.type,j+1)][:] = dbatch.label[j]

		executor.forward(is_train=True)
		executor.backward()

		for index, key in enumerate(keys):
			if 'img' not in key and 'stereo' not in key and 'flow' not in key:
				opt.update(index, executor.arg_dict[key], grads[key], states[key])

		endpointerr.update(executor.outputs[0],executor.arg_dict['{}_downsample1'.format(cmd.type)])
		logging.info('iteration :  {} th  Average End Point Error : {} '.format(iteration,endpointerr.get()))

	cmd.con += 1
	mx.model.save_checkpoint( cfg.MODEL.checkpoint_prefix + cmd.type,cmd.con,net,executor.arg_dict,executor.arg_dict)
	logging.info('epoch  : {} th : Average End point Error : {} '.format(i+1,endpointerr.get()[1]))

