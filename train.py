import argparse
import logging
import mxnet as mx
import dataiter
import dataset
import metric
import numpy as np

from config import cfg, batchsize
from symbol import dispnet_symbol, enet_symbol, rnn_symbol, \
    spynet_symbol, res_symbol,cnn_symbol,stereo_rnn_symbol,\
    spynet2_symbol,spynet3_symbol,flownet_simple_symbol
from utils import util

if __name__ == '__main__':

    # logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')

    # parse parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue', action='store', dest='con', type=int, default=0, help='begin epoch of training')
    parser.add_argument('--lr', action='store', dest='lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--model', action='store', dest='model', type=str,default='dispnet',help='choose symbol',
        				choices=['dispnet', 'enet', 'rnn', 'spynet1', 'resnet','enet2','enet3','spynet2','spynet3','flownets'])
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--dataset', type=str, default='synthesis', help='choose dataset',
        				choices=['flyingthing', 'flyingchair', 'kitti', 'tusimple', 'flyingthing1000',
                                 'flyingchair1000','multidataset'])
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
    num_hidden = None
    fixed_param_names=None
    label_sparse = True if  cmd.dataset == 'kitti' or cmd.dataset == 'tusimple' or cmd.dataset=='multidataset' else False
    lr_mult = None
    print label_sparse
    # choose symbol and load args
    if cmd.model == 'dispnet':
        batchsize = (2, 3, 320, 1152)
        label_shape = [(batchsize[2] / (2**i), batchsize[3] / (2**i)) for i in range(0, 1)]
        net = dispnet_symbol.stereo_net(cmd.type, label_sparse)
        # fixed_param_names = [item for item in net.list_arguments() if 'drr' not in item]
        # fixed_param_names.extend(item for item in net.list_arguments() if 'init_label.drr' in item)
        # lr_mult = {item :1.0 for item in fixed_param_names}
        # fixed_param_names
        print fixed_param_names

    elif cmd.model == 'resnet':
        batchsize = (4, 3, 320, 1024)
        label_shape = [(batchsize[2] / (2**i), batchsize[3] / (2**i)) for i in range(0, 3)]
        net = res_symbol.stereo_net(label_sparse)
        # fixed_param_names = [item for item in net.list_arguments() if 'drr' not in item]
        # fixed_param_names.extend(item for item in net.list_arguments() if 'init_label.drr' in item)

    elif cmd.model == 'flownets':

        batchsize = (4, 3, 320, 512)
        label_shape = [(batchsize[2] / (2 ** i), batchsize[3] / (2 ** i)) for i in range(1, 3)]
        net = flownet_simple_symbol.flownet_simple(cmd.type,label_sparse)

    elif cmd.model == 'enet':
        batchsize = (8, 3, 384, 768)
        label_shape = [(192, 384), (192 / 2, 384 / 2), (192 / 4, 384 / 4)]
        net = enet_symbol.get_body4(bn_momentum=0.90, is_sparse=label_sparse)

    elif cmd.model == 'enet2':
        batchsize = (4, 3, 384, 768)
        label_shape = [(192, 384), (192 / 2, 384 / 2), (192 / 4, 384 / 4)]
        net = enet_symbol.get_body3(bn_momentum=0.90, is_sparse=label_sparse)

    elif cmd.model == 'enet3':
        batchsize = (4, 3, 384, 768)
        label_shape = [(192, 384), (192 / 2, 384 / 2), (192 / 4, 384 / 4)]
        net = enet_symbol.get_body2(bn_momentum=0.90, is_sparse=label_sparse)

    elif cmd.model == 'rnn':
        batchsize = (70, 3, 32, 256)
        label_shape = [(16, 128)]
        use_rnn = True
        num_hidden = 64
        net = stereo_rnn_symbol.rnn(batchsize[2], batchsize[3], num_hidden)
        # fixed_param_names= [item for item in net.list_arguments() if 'share' in item]

    elif cmd.model == 'spynet1':
        batchsize = (10, 3, 320, 1216)
        label_shape = [(320 / (2**i), 1216 / (2**i)) for i in range(0, 5)]
        net = spynet_symbol.spynet_symbol(type=cmd.type)

    elif cmd.model == 'spynet2':
        batchsize = (4, 3, 384, 512)
        label_shape = [(384 / (2 ** i), 512 / (2 ** i)) for i in range(6)]
        net = spynet2_symbol.stereo_net(net_type=cmd.type)

    elif cmd.model == 'spynet3':
        batchsize = (4, 3, 384, 512)
        label_shape = [(384 / (2 ** i), 512 / (2 ** i)) for i in range(0, 7)]
        net = spynet3_symbol.spynet_symbol(type=cmd.type)
        # fixed_param_names = [item for item in net.list_arguments() if 'v5' not in item]

    if cmd.con == 0:
        if cmd.model =='rnn':
            _, args, auxs = mx.model.load_checkpoint(cfg.MODEL.stereomatching_checkpoint, 18)
        else:
            args = None
            auxs = None
    else:
        args, auxs = util.load_checkpoint(
            cfg.MODEL.checkpoint_prefix + cmd.type + '_' + cmd.model, cmd.con)
        logging.info("load the {} th epoch paramaters".format(cmd.con))

    mx.model.save_checkpoint('/home/xudong/model_zoo/dispnet_drr', 0, net, args, auxs)
    # dataset and iterator
    if cmd.dataset == 'kitti':
        # data_set = dataset.KittiDataset(cmd.type, '2015')
        # data_set.dirs.extend(dataset.KittiDataset(cmd.type, '2012').dirs[:100])
        data_set = dataset.KittiDataset(cmd.type, '2015')
        data_set.dirs.extend(data_set.dirs)
        data_set.dirs.extend(data_set.dirs)
        data_set.dirs.extend(data_set.dirs)
        # data_set.dirs = data_set.dirs[:160]
        data_set.dirs.extend(dataset.KittiDataset(cmd.type, '2012').dirs)
        # data_set.dirs.extend(data_set.dirs)
        # data_set.dirs.extend(data_set.dirs)
        # data_set.dirs.extend(data_set.dirs)
        # data_set.dirs.extend(data_set.dirs)
        # data_set.dirs.extend(data_set.dirs)
        input_shape = (350, 1200)
    elif cmd.dataset == 'tusimple':
        data_set = dataset.TusimpleDataset(num_data=4000)
        input_shape = data_set.shapes()
    elif cmd.dataset == 'flyingthing':
        data_set = dataset.SythesisData(cmd.type, ['flyingthing3d','Driving','Monkaa'])
        input_shape = data_set.shapes()
    elif cmd.dataset == 'flyingchair':
        data_set = dataset.multidataset(cmd.type)
        data_set.add_dataset(dataset.FlyingChairsDataset())
        # data_set.add_dataset(dataset.KittiDataset(cmd.type, '2015'))
        # data_set.add_dataset(dataset.KittiDataset(cmd.type, '2012'))
        input_shape = data_set.shapes()
        print len(data_set.dirs)
    elif cmd.dataset == 'flyingchair1000':
        data_set = dataset.FlyingChairsDataset()
        data_set.dirs = data_set.dirs[:1000]
        data_set.dirs.extend(data_set.dirs)
        input_shape = data_set.shapes()
    elif cmd.dataset == 'flyingthing1000':
        data_set = dataset.SythesisData(cmd.type, ['flyingthing3d'])
        data_set.dirs = data_set.dirs[:1000]
        input_shape = data_set.shapes()
    elif cmd.dataset == 'multidataset':
        data_set = dataset.multidataset(cmd.type)
        kitti = dataset.KittiDataset(cmd.type, '2015')
        kitti.dirs = kitti.dirs[:160]
        kitti.dirs.extend(kitti.dirs)
        kitti.dirs.extend(kitti.dirs)
        kitti.dirs.extend(kitti.dirs)
        kitti.dirs.extend(kitti.dirs)
        kitti.dirs.extend(kitti.dirs)
        kitti.dirs.extend(kitti.dirs)
        kitti.dirs.extend(kitti.dirs)
        # kitti.dirs.extend(kitti.dirs)
        data_set.add_dataset(kitti)
        synthesis = dataset.SythesisData(cmd.type, ['flyingthing3d'])
        synthesis.dirs = synthesis.dirs[:100]
        # data_set.add_dataset(synthesis)
        # data_set.add_dataset(dataset.TusimpleDataset(num_data=4000))
        input_shape = (350, 900)
        print len(data_set.dirs)
    else:
        raise ValueError('the dataset do not exist')
    print '??'
    if cmd.iterator == 'caffeiter':
        data = dataiter.caffe_iterator(ctx, data_set, batchsize, label_shape, input_shape,
            n_thread=cmd.thread, use_rnn=use_rnn, data_type=cmd.type)

    elif cmd.iterator == 'record':
        records = util.get_imageRecord(dataset.SythesisData(cmd.type, ['flyingthing3d']), batchsize[0], cmd.thread)
        data = dataiter.multi_imageRecord(
            records=records,
            data_type=cmd.type,
            batch_shape=batchsize,
            label_shapes=label_shape,
            augment_ratio=0.0,
            use_rnn=use_rnn)

    elif cmd.iterator == 'pythoniter':
        data = dataiter.Dataiter_training(
            dataset=data_set,
            batch_shape=batchsize,
            label_shape=label_shape,
            augment_ratio=0.0,
            n_thread=cmd.thread,
            be_shuffle = True,
            is_bilinear = True,
            use_rnn=use_rnn,
            num_hidden=num_hidden)

    eval_data = dataset.KittiDataset(cmd.type, '2012')
    # eval_data.dirs = eval_data.dirs[160:]
    eval_data.dirs.extend(eval_data.dirs)
    eval_dataiter = dataiter.Dataiter_training(
            dataset=eval_data,
            batch_shape=batchsize,
            label_shape=label_shape,
            augment_ratio=0.0,
            n_thread=cmd.thread,
            be_shuffle = True,
            is_bilinear = False,
            use_rnn=use_rnn,
            num_hidden=num_hidden)
    eval_dataiter.reset()
    # eval_dataiter = None
    data.reset()
    # data = dataiter.DummyIter(data)
    # metric
    err = [metric.EndPointErr(cmd.which_scale)]
    if cmd.type =='stereo':
        err.append(metric.D1all())

    # optimizer
    if cmd.optimizer == 'adam':
        optimizer = util.Adam(learning_rate=cmd.lr, beta1=cfg.ADAM.beta1, beta2=cfg.ADAM.beta2,
                              rescale_grad=1.0 / batchsize[0], epsilon=cfg.ADAM.epsilon, wd=cfg.ADAM.weight_decay,
                              clip_gradient=1.0)
    else:
        optimizer = mx.optimizer.SGD(learning_rate=cmd.lr, momentum=0.90, wd=cfg.ADAM.weight_decay,
                                     rescale_grad=1.0 / batchsize[0],clip_gradient=1.0)
    optimizer.idx2name = util.get_idx2name(net)
    if lr_mult is not None:
        optimizer.set_lr_mult(lr_mult)
    init = mx.initializer.Xavier(rnd_type='gaussian', factor_type='in',
                                 magnitude=cfg.MODEL.weight_init_scale)
    # init = mx.initializer.Orthogonal(scale=0.1, rand_type='uniform')
    # train
    mod = mx.module.Module(symbol=net,
                           data_names=[item[0] for item in data.provide_data],
                           label_names=[item[0] for item in data.provide_label],
                           context=ctx,
                           fixed_param_names=fixed_param_names)

    mod.fit(train_data=data, eval_metric=err,
            eval_data=eval_dataiter,
            epoch_end_callback=mx.callback.do_checkpoint(cfg.MODEL.checkpoint_prefix + cmd.type + '_' + cmd.model),
            batch_end_callback=[mx.callback.Speedometer(batchsize[0], 10)],
            kvstore='device',
            optimizer=optimizer,
            initializer= init,
            arg_params=args,
            aux_params=auxs,
            begin_epoch=cmd.con,
            num_epoch=cfg.MODEL.epoch_num,
            allow_missing=True)
