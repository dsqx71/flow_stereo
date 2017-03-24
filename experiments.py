"""
In machine learing experiments, "Hard code" is inevitable.  You should carefully review the code below and
corresponding symbol.

The design purpose of experiments.py is to isolate "hard code" of experiments from reset components, and help
users strictly perform experiments.
"""
import shutil
import os
import mxnet as mx
from symbol import dispnet_symbol
from data import dataset, dataloader, augmentation, config
from others import util, metric

experiments = []
def register(f):
    global experiments
    if f.__name__ in experiments:
        raise ValueError("{} already exists".format(f.__name__))
    else:
        experiments.append(f.__name__)
    return f

@register
def dispnet_finetune(epoch, ctx, lr):
    """
    Load caffe pretrain model, and fine tune dispnet on KITTI 2015 stereo dataset
    """

    # model name
    experiment_name = 'dispnet_finetune'
    pretrain_model = 'caffe_pretrain'

    # shapes
    batch_shape = (4, 3, 320, 768)
    num_iteration = 80000

    # optimizer params
    optimizer_type = 'Adam'
    optimizer_setting = dict(learning_rate = lr,
                             beta1 = 0.90,
                             beta2 = 0.999,
                             epsilon = 1e-4,
                             wd = 0.0004)
    # symbol params
    loss_scale = {'loss1': 1.00,
                  'loss2': 0.00,
                  'loss3': 0.00,
                  'loss4': 0.00,
                  'loss5': 0.00,
                  'loss6': 0.00}

    net = dispnet_symbol.dispnet(loss_scale=loss_scale, net_type='stereo', is_sparse=True)

    # dataset
    data_set = dataset.KittiDataset('stereo', '2015', is_train=True)

    # augmentation setting
    augment_pipeline = augmentation.augmentation(max_num_tries=30,
                                                cropped_height=batch_shape[2],
                                                cropped_width=batch_shape[3],
                                                data_type='stereo',
                                                augment_ratio=1.0,
                                                noise_std=0.01,
                                                mirror_rate=0.0,
                                                rotate_range={'method': 'uniform', 'low': 0, 'high': 0},
                                                translate_range={'method': 'uniform', 'low': -0.2, 'high': 0.2},
                                                zoom_range={'method': 'normal', 'mean': 1.0, 'scale': 0.5},
                                                squeeze_range={'method': 'normal', 'mean': 1.0, 'scale': 0.5},
                                                gamma_range={'method': 'normal', 'mean': 1.0, 'scale': 0.3},
                                                brightness_range={'method': 'normal', 'mean': 0.0, 'scale': 0.005},
                                                contrast_range={'method': 'normal', 'mean': 1.0, 'scale': 0.6},
                                                rgb_multiply_range={'method': 'normal', 'mean': 1.0, 'scale': 0.5},
                                                interpolation_method='nearest')

    # metric
    eval_metric = [metric.D1all(), metric.EndPointErr()]

    # initializer
    init = mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=8)

    # load pretrain model
    checkpoint_prefix = os.path.join(config.cfg.model.check_point, experiment_name)

    # create dir
    if os.path.isdir(checkpoint_prefix) == False:
        os.makedirs(checkpoint_prefix)

    if epoch == 0:
        # caffe pretrain model
        pretrain_model_path = os.path.join(config.cfg.model.pretrain_model_prefix, pretrain_model)
        args, auxs = util.load_checkpoint(pretrain_model_path, 0)
    else:
        # previous training checkpoint
        checkpoint_path = os.path.join(checkpoint_prefix, experiment_name)
        args, auxs = util.load_checkpoint(checkpoint_path, epoch)

    # infer shapes of outputs
    shapes = net.infer_shape(img1=batch_shape, img2=batch_shape)
    label_shape = zip(net.list_outputs(), shapes[1])

    # data loader
    dataiter = dataloader.numpyloader(experiment_name= experiment_name,
                                      dataset = data_set,
                                      augmentation= augment_pipeline,
                                      batch_shape=batch_shape,
                                      label_shape= label_shape,
                                      n_thread=3,
                                      half_life=10000,
                                      initial_coeff=0.7,
                                      final_coeff=1.0,
                                      interpolation_method='nearest')

    # module
    mod = mx.module.Module(symbol=net,
                           data_names=[item[0] for item in dataiter.provide_data],
                           label_names=[item[0] for item in dataiter.provide_label],
                           context=ctx,
                           fixed_param_names=None)
    # training
    mod.fit(train_data=dataiter,
            eval_metric=eval_metric,
            epoch_end_callback=mx.callback.module_checkpoint(mod,
                                                             checkpoint_prefix+experiment_name,
                                                             period=50,
                                                             save_optimizer_states=True),
            batch_end_callback=[mx.callback.Speedometer(batch_shape[0], 10)],
            kvstore='device',
            optimizer=optimizer_type,
            optimizer_params= optimizer_setting,
            initializer=init,
            arg_params=args,
            aux_params=auxs,
            begin_epoch=epoch,
            num_epoch= int(num_iteration/(dataiter.data_num/batch_shape[0])))

    # save reuslt
    # json cannot save CustomOp
    net_saved = dispnet_symbol.dispnet(loss_scale=loss_scale, net_type='stereo', is_sparse=False)
    args, auxs = mod.get_params()
    model_zoo_path = os.path.join(config.cfg.model.model_zoo, experiment_name)
    mx.model.save_checkpoint(prefix=model_zoo_path,
                             epoch=0,
                             symbol=net_saved,
                             arg_params=args,
                             aux_params=auxs)
    # copy mean file to model zoo directory
    shutil.copy2(os.path.join(config.cfg.dataset.mean_dir, experiment_name + '_mean.npy'), config.cfg.model.model_zoo)

@register
def dispnet_pretrain(epoch, ctx, lr):
    """
      train dispnet using flyingthing dataset
    """
    # model name
    experiment_name = 'dispnet_pretrain'

    # shapes
    batch_shape = (4, 3, 384, 768)
    num_iteration = 1400000

    # optimizer params
    optimizer_type = 'Adam'
    optimizer_setting = dict(learning_rate=lr,
                             beta1=0.90,
                             beta2=0.999,
                             epsilon=1e-4,
                             wd=0.0004,
                             rescale_grad = 1.0/batch_shape[0],
                             lr_scheduler=mx.lr_scheduler.FactorScheduler(step=200000,
                                                                          factor=0.5,
                                                                          stop_factor_lr=3.125E-6))

    # symbol params
    loss_scale = {'loss1': 1.00,
                  'loss2': 0.00,
                  'loss3': 0.00,
                  'loss4': 0.00,
                  'loss5': 0.00,
                  'loss6': 0.00}

    net = dispnet_symbol.dispnet(loss_scale=loss_scale, net_type='stereo', is_sparse=False)

    # dataset
    data_set = dataset.SynthesisData(data_type='stereo',
                                     scene_list=['flyingthing3d'],
                                     rendering_level=['cleanpass'])

    # augmentation setting
    augment_pipeline = augmentation.augmentation(max_num_tries=1,
                                                 cropped_height=batch_shape[2],
                                                 cropped_width=batch_shape[3],
                                                 data_type='stereo',
                                                 augment_ratio=1.0,
                                                 noise_std=0.005,
                                                 mirror_rate=0.0,
                                                 rotate_range={'method': 'uniform', 'low': 0, 'high': 0},
                                                 translate_range={'method': 'normal', 'mean': 0.0, 'scale': 0.4},
                                                 zoom_range={'method': 'normal', 'mean': 1.1, 'scale': 0.3},
                                                 squeeze_range={'method': 'normal', 'mean': 1.0, 'scale': 0.3},
                                                 gamma_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},
                                                 brightness_range={'method': 'normal', 'mean': 0.0, 'scale': 0.01},
                                                 contrast_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},
                                                 rgb_multiply_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},
                                                 interpolation_method='bilinear')

    # metric
    eval_metric = [metric.EndPointErr()]

    # initializer
    init = mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=8)

    # pretrain model
    checkpoint_prefix = os.path.join(config.cfg.model.check_point, experiment_name)
    checkpoint_path = os.path.join(checkpoint_prefix, experiment_name)
    # create dir
    if os.path.isdir(checkpoint_prefix) == False:
        os.makedirs(checkpoint_prefix)

    if epoch == 0:
        args = None
        auxs = None
    else:
        # previous training checkpoint
        args, auxs = util.load_checkpoint(checkpoint_path, epoch)

    # infer shapes of outputs
    shapes = net.infer_shape(img1=batch_shape, img2=batch_shape)
    label_shape = zip(net.list_outputs(), shapes[1])

    # data loader
    dataiter = dataloader.numpyloader(experiment_name=experiment_name,
                                      dataset=data_set,
                                      augmentation=augment_pipeline,
                                      batch_shape=batch_shape,
                                      label_shape=label_shape,
                                      n_thread=5,
                                      half_life=200000,
                                      initial_coeff=0.5,
                                      final_coeff=1.0,
                                      interpolation_method='bilinear')

    # module
    mod = mx.module.Module(symbol=net,
                           data_names=[item[0] for item in dataiter.provide_data],
                           label_names=[item[0] for item in dataiter.provide_label],
                           context=ctx,
                           fixed_param_names=None)
    # training
    mod.fit(train_data=dataiter,
            eval_metric=eval_metric,
            epoch_end_callback=mx.callback.module_checkpoint(mod,
                                                             checkpoint_path,
                                                             period=2,
                                                             save_optimizer_states=True),
            batch_end_callback=[mx.callback.Speedometer(batch_shape[0], 20)],
            kvstore='device',
            optimizer=optimizer_type,
            optimizer_params=optimizer_setting,
            initializer=init,
            arg_params=args,
            aux_params=auxs,
            begin_epoch=epoch,
            num_epoch=300)
            #int(num_iteration / (dataiter.data_num / batch_shape[0])))

    # save reuslt
    # cannot save CustomOp into json file
    net_saved = dispnet_symbol.dispnet(loss_scale=loss_scale, net_type='stereo', is_sparse=False)
    args, auxs = mod.get_params()
    model_zoo_path = os.path.join(config.cfg.model.model_zoo, experiment_name)
    mx.model.save_checkpoint(prefix=model_zoo_path,
                             epoch=0,
                             symbol=net_saved,
                             arg_params=args,
                             aux_params=auxs)

    # copy mean to model zoo directory
    shutil.copy2(os.path.join(config.cfg.dataset.mean_dir, experiment_name+'_mean.npy'), config.cfg.model.model_zoo)

@register
def flownet_pretrain(epoch, ctx, lr):
    """
      train flownet on flyingchair dataset
    """
    # model name
    experiment_name = 'flownet_pretrain'
    data_type = 'flow'

    # shapes
    batch_shape = (8, 3, 384, 448)
    num_iteration = 750000

    # optimizer params
    optimizer_type = 'Adam'
    optimizer_setting = dict(learning_rate=lr,
                             beta1=0.90,
                             beta2=0.999,
                             epsilon=1e-4,
                             rescale_grad=1.0/batch_shape[0],
                             wd=0.00001,
                             lr_scheduler=mx.lr_scheduler.FactorScheduler(step=250000,
                                                                          factor=0.5,
                                                                          stop_factor_lr=3.125E-6))

    # symbol params
    loss_scale = {'loss1': 1.00,
                  'loss2': 0.00,
                  'loss3': 0.00,
                  'loss4': 0.00,
                  'loss5': 0.00,
                  'loss6': 0.00}

    net = dispnet_symbol.dispnet(loss_scale=loss_scale, net_type=data_type, is_sparse=False)

    # dataset
    data_set = dataset.FlyingChairsDataset()
    # augmentation setting
    augment_pipeline = augmentation.augmentation(max_num_tries=30,
                                                 cropped_height=batch_shape[2],
                                                 cropped_width=batch_shape[3],
                                                 data_type=data_type,
                                                 augment_ratio=1.0,
                                                 noise_std=0.001,
                                                 mirror_rate=0.0,
                                                 rotate_range={'method': 'uniform', 'low': -17, 'high': 17},
                                                 translate_range={'method': 'uniform', 'low': -0.2, 'high': 0.2},
                                                 zoom_range={'method': 'normal', 'mean': 1.0, 'scale': 0.1},
                                                 squeeze_range={'method': 'normal', 'mean': 1.0, 'scale': 0.1},
                                                 gamma_range={'method': 'normal', 'mean': 1.0, 'scale': 0.1},
                                                 brightness_range={'method': 'normal', 'mean': 0.0, 'scale': 0.001},
                                                 contrast_range={'method': 'normal', 'mean': 1.0, 'scale': 0.1},
                                                 rgb_multiply_range={'method': 'normal', 'mean': 1.0, 'scale': 0.1},
                                                 interpolation_method='bilinear')

    # metric
    eval_metric = [metric.EndPointErr()]

    # initializer
    init = mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=8)

    # pretrain model
    checkpoint_prefix = os.path.join(config.cfg.model.check_point, experiment_name)
    checkpoint_path = os.path.join(checkpoint_prefix, experiment_name)
    # create dir
    if os.path.isdir(checkpoint_prefix) == False:
        os.makedirs(checkpoint_prefix)

    if epoch == 0:
        args = None
        auxs = None
    else:
        # previous training checkpoint
        args, auxs = util.load_checkpoint(checkpoint_path, epoch)

    # infer shapes of outputs
    shapes = net.infer_shape(img1=batch_shape, img2=batch_shape)
    label_shape = zip(net.list_outputs(), shapes[1])

    # data loader
    dataiter = dataloader.numpyloader(experiment_name=experiment_name,
                                      dataset=data_set,
                                      augmentation=augment_pipeline,
                                      batch_shape=batch_shape,
                                      label_shape=label_shape,
                                      n_thread=8,
                                      half_life=100000,
                                      initial_coeff=0.5,
                                      final_coeff=1.0,
                                      interpolation_method='bilinear')

    # module
    mod = mx.module.Module(symbol=net,
                           data_names=[item[0] for item in dataiter.provide_data],
                           label_names=[item[0] for item in dataiter.provide_label],
                           context=ctx,
                           fixed_param_names=None)
    # training
    mod.fit(train_data=dataiter,
            eval_metric=eval_metric,
            epoch_end_callback=mx.callback.module_checkpoint(mod,
                                                             checkpoint_path,
                                                             period=10,
                                                             save_optimizer_states=True),
            batch_end_callback=[mx.callback.Speedometer(batch_shape[0], 20)],
            kvstore='device',
            optimizer=optimizer_type,
            optimizer_params=optimizer_setting,
            initializer=init,
            arg_params=args,
            aux_params=auxs,
            begin_epoch=epoch,
            num_epoch=int(num_iteration / (dataiter.data_num / batch_shape[0])))

    # save reuslt
    # json cannot save CustomOp
    net_saved = dispnet_symbol.dispnet(loss_scale=loss_scale, net_type=data_type, is_sparse=False)
    args, auxs = mod.get_params()
    model_zoo_path = os.path.join(config.cfg.model.model_zoo, experiment_name)
    mx.model.save_checkpoint(prefix=model_zoo_path,
                             epoch=0,
                             symbol=net_saved,
                             arg_params=args,
                             aux_params=auxs)
    # copy mean file to model zoo directory
    shutil.copy2(os.path.join(config.cfg.dataset.mean_dir, experiment_name + '_mean.npy'), config.cfg.model.model_zoo)