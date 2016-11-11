import argparse
import logging
import mxnet as mx
import dataiter
import dataset
import metric
from config import cfg, batchsize
from symbol import dispnet_symbol, enet_symbol, rnn_symbol, spynet_symbol
from utils import util

if __name__ == '__main__':

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')

    # parse parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue', action='store', dest='con', type=int, default=0, help='begin epoch of training')
    parser.add_argument('--lr', action='store', dest='lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--model', action='store', dest='model', type=str,default='dispnet',help='choose symbol',
        				choices=['dispnet', 'enet', 'rnn', 'spynet'])
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--dataset', type=str, default='synthesis', help='choose dataset',
        				choices=['synthesis', 'flyingchair', 'kitti', 'tusimple'])
    parser.add_argument('--thread', type=int, default=30, help='number of thread in iterator')
    parser.add_argument('--type', type=str, default='stereo',
        				choices=['stereo', 'flow'], help='stereo or optical flow')
    parser.add_argument('--iterator', type=str, default='caffeiter',
						help=' there are three iterator. caffeiter and record only support stereo, pythoniter support both of them',
						choices=['caffeiter', 'pythoniter', 'record'])
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--which_scale', type=int, default=0, help='which scale does metric consider')
    cmd = parser.parse_args()

    ctx = [mx.gpu(int(i)) for i in cmd.gpus.split(',')]
    use_rnn = False
    label_sparse = True if cmd.dataset == 'kitti' or cmd.dataset == 'tusimple' else False

    # choose symbol and  load args

    if cmd.model == 'dispnet':
        batchsize = (1, 3, 320, 1216)
        label_shape = [(160, 608)]
        net = dispnet_symbol.stereo_net(label_sparse)

        print net.list_outputs()
    elif cmd.model == 'enet':
        batchsize = (2, 3, 320, 768)
        label_shape = [(160, 384)]
        net = enet_symbol.get_body(bn_momentum=0.90, is_sparse=label_sparse)

    elif cmd.model == 'enet2':
        batchsize = (4, 3, 384, 768)
        label_shape = [(384, 768)]
        net = enet_symbol.get_body2(bn_momentum=0.90)

    elif cmd.model == 'enet4':
        batchsize = (2, 3, 384, 768)
        label_shape = [(192, 384)]
        net = enet_symbol.get_body3(bn_momentum=0.90)

    elif cmd.model == 'rnn':
        batchsize = (4, 3, 384, 768)
        label_shape = [(192, 384)]
        use_rnn = True
        net = rnn_symbol.rnn(
            bn_momentum=0.90,
            height=batchsize[2],
            width=batchsize[3])

    elif cmd.model == 'spynet':
        batchsize = (8, 3, 384, 512)
        label_shape = [(384 / (2**i), 512 / (2**i)) for i in range(6)]
        net = spynet_symbol.spynet_symbol(height=384, width=512)

    if cmd.con == 0:
        args = None
        auxs = None

    else:
        args, auxs = util.load_checkpoint(
            cfg.MODEL.checkpoint_prefix + cmd.type + '_' + cmd.model, cmd.con, net, batchsize)
        logging.info("load the {} th epoch paramaters".format(cmd.con))

    # dataset and iterator
    if cmd.dataset == 'kitti':
        data_set = dataset.KittiDataset(cmd.type, '2015')
        input_shape = (350, 1200)
    elif cmd.dataset == 'tusimple':
        data_set = dataset.TusimpleDataset(num_data=4000)
        input_shape = data_set.shapes()
    elif cmd.dataset == 'synthesis':
        data_set = dataset.SythesisData(cmd.type, ['flyingthing3d'])
        input_shape = data_set.shapes()
    elif cmd.dataset == 'flyingchair':
        data_set = dataset.FlyingChairsDataset()
        input_shape = data_set.shapes()
    else:
        raise ValueError('the dataset do not exist')

    if cmd.iterator == 'caffeiter':
        data = dataiter.caffe_iterator(ctx, data_set, batchsize, label_shape, input_shape,
            n_thread=cmd.thread, use_rnn=use_rnn)

    elif cmd.iterator == 'record':
        records = util.get_imageRecord(dataset.SythesisData(cmd.type, ['flyingthing3d']), batchsize[0], cmd.thread)
        data = dataiter.multi_imageRecord(
            records=records,
            data_type=cmd.type,
            batch_shape=batchsize,
            label_shapes=label_shape,
            augment_ratio=0,
            use_rnn=use_rnn)

    elif cmd.iterator == 'pythoniter':
        data = dataiter.Dataiter_training(
            dataset=data_set,
            batch_shape=batchsize,
            label_shape=label_shape,
            augment_ratio=0,
            n_thread=cmd.thread,
            be_shuffle=True,
            downsample_method='interpolate',
            use_rnn=use_rnn)

    # metric
    err = metric.EndPointErr(cmd.which_scale)

    # optimizer
    if cmd.optimizer == 'adam':
        optimizer = util.Adam(learning_rate=cmd.lr, beta1=cfg.ADAM.beta1, beta2=cfg.ADAM.beta2,
                              rescale_grad=1.0 / batchsize[0], epsilon=cfg.ADAM.epsilon, wd=cfg.ADAM.weight_decay,
                              clip_gradient=1000000, num_ctx=len(ctx))
    else:
        optimizer = mx.optimizer.SGD(learning_rate=cmd.lr, momentum=0.90, wd=cfg.ADAM.weight_decay,
                                     rescale_grad=1.0 / batchsize[0])
    optimizer.idx2name = util.get_idx2name(net)

    # train
    model = mx.model.FeedForward(
            ctx=ctx,
            symbol=net,
            num_epoch=cfg.MODEL.epoch_num,
            optimizer=optimizer,
            initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=cfg.MODEL.weight_init_scale),
            arg_params=args,
            aux_params=auxs,
            begin_epoch=cmd.con,
            lr_scheduler=mx.lr_scheduler.FactorScheduler(10 * 30000 / batchsize[0], 0.8))

    model.fit(X=data, eval_metric=err,
             epoch_end_callback=mx.callback.do_checkpoint(
                 cfg.MODEL.checkpoint_prefix + cmd.type + '_' + cmd.model),
             batch_end_callback=[mx.callback.Speedometer(batchsize[0],10)],
             kvstore='local_allreduce_device',
             work_load_list=None)
