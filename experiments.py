"""
In machine learing experiments, "Hard code" is inevitable.  You should carefully review the code below and
corresponding symbol.

The design purpose of experiments.py is to isolate "hard code" of experiments from reset components, and help
users strictly perform experiments.
"""
import shutil
import os
import mxnet as mx
from symbol import dispnet_symbol, resnet_symbol, spynet_symbol, \
                   flownet2_symbol, DRR_symbol, flownets_half_symbol, \
                   flownet2ss_symbol, dispnet2CSS_symbol, flownet2CSS_origin_symbol
from data import dataset, dataloader, augmentation, config, LMDB
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
    augment_pipeline = augmentation.augmentation(
        interpolation_method='bilinear',
        max_num_tries=10,
        cropped_height=384,
        cropped_width=768,
        data_type='stereo',
        augment_ratio=0.9,
        noise_std=0.05,
        mirror_rate=0.0,
        flip_rate=0.1,
        rotate_range={'method': 'uniform', 'low': 0, 'high': 0},
        translate_range={'method': 'normal', 'mean': 0.0, 'scale': 0.4},
        zoom_range={'method': 'normal', 'mean': 1.0, 'scale': 0.2},
        squeeze_range={'method': 'normal', 'mean': 1.0, 'scale': 0.2},
        gamma_range={'method': 'normal', 'mean': 1.0, 'scale': 0.03},
        brightness_range={'method': 'normal', 'mean': 0.0, 'scale': 0.06},
        contrast_range={'method': 'normal', 'mean': 1.0, 'scale': 0.06},
        rgb_multiply_range={'method': 'normal', 'mean': 1.0, 'scale': 0.06},

        lmult_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        lmult_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        lmult_add={'method': 'normal', 'mean': 0.00, 'scale': 0.06},

        sat_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        sat_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        sat_add={'method': 'normal', 'mean': 0.00, 'scale': 0.06},

        col_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        col_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        col_add={'method': 'normal', 'mean': 0.00, 'scale': 0.06},

        ladd_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        ladd_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        ladd_add={'method': 'normal', 'mean': 0.00, 'scale': 0.06},
        col_rotate={'method': 'normal', 'mean': 0, 'scale': 6})

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
    tmp = zip(net.list_arguments(), shapes[0])
    label_shape = [item for item in tmp if 'label' in item[0]]
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
    augment_pipeline = augmentation.augmentation(
        interpolation_method='bilinear',
        max_num_tries=10,
        cropped_height=384,
        cropped_width=768,
        data_type='stereo',
        augment_ratio=0.9,
        noise_std=0.05,
        mirror_rate=0.0,
        flip_rate=0.1,
        rotate_range={'method': 'uniform', 'low': 0, 'high': 0},
        translate_range={'method': 'normal', 'mean': 0.0, 'scale': 0.4},
        zoom_range={'method': 'normal', 'mean': 1.0, 'scale': 0.2},
        squeeze_range={'method': 'normal', 'mean': 1.0, 'scale': 0.2},
        gamma_range={'method': 'normal', 'mean': 1.0, 'scale': 0.03},
        brightness_range={'method': 'normal', 'mean': 0.0, 'scale': 0.06},
        contrast_range={'method': 'normal', 'mean': 1.0, 'scale': 0.06},
        rgb_multiply_range={'method': 'normal', 'mean': 1.0, 'scale': 0.06},

        lmult_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        lmult_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        lmult_add={'method': 'normal', 'mean': 0.00, 'scale': 0.06},

        sat_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        sat_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        sat_add={'method': 'normal', 'mean': 0.00, 'scale': 0.06},

        col_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        col_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        col_add={'method': 'normal', 'mean': 0.00, 'scale': 0.06},

        ladd_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        ladd_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.6},
        ladd_add={'method': 'normal', 'mean': 0.00, 'scale': 0.06},
        col_rotate={'method': 'normal', 'mean': 0, 'scale': 6})

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
    tmp = zip(net.list_arguments(), shapes[0])
    label_shape = [item for item in tmp if 'label' in item[0]]

    # training data loader
    dataiter = dataloader.numpyloader(experiment_name=experiment_name,
                                      dataset=data_set,
                                      augmentation=augment_pipeline,
                                      batch_shape=batch_shape,
                                      label_shape=label_shape,
                                      n_thread=6,
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
            num_epoch=680)
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
    batch_shape = (8, 3, 384, 512)
    num_iteration = 1200000

    # optimizer params
    optimizer_type = 'Adam'
    optimizer_setting = dict(learning_rate=lr,
                             beta1=0.90,
                             beta2=0.999,
                             epsilon=1e-4,
                             rescale_grad=1.0/batch_shape[0],
                             wd=0.0004,
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
    # data_set = dataset.FlyingChairsDataset()
    data_set = dataset.SynthesisData(data_type='flow',
                                     scene_list=['flyingthing3d'],
                                     rendering_level=['cleanpass'])
    # augmentation setting
    augment_pipeline = augmentation.augmentation(
            interpolation_method='bilinear',
            max_num_tries=10,
            cropped_height=384,
            cropped_width=512,
            data_type='stereo',
            augment_ratio=1.0,
            noise_std=0.03,
            mirror_rate=0.0,
            flip_rate=0.1,
            rotate_range={'method': 'uniform', 'low': -17, 'high': 17},
            translate_range={'method': 'normal', 'mean': 0.0, 'scale': 0.4},
            zoom_range={'method': 'normal', 'mean': 1.0, 'scale': 0.3},
            squeeze_range={'method': 'normal', 'mean': 1.0, 'scale': 0.3},
            gamma_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},
            brightness_range={'method': 'normal', 'mean': 0.0, 'scale': 0.01},
            contrast_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},
            rgb_multiply_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},

            # refer to Dispnet Caffe code
            lmult_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            lmult_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            lmult_add={'method': 'normal', 'mean': 0.00, 'scale': 0.03},

            sat_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            sat_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            sat_add={'method': 'normal', 'mean': 0.00, 'scale': 0.03},

            col_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            col_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            col_add={'method': 'normal', 'mean': 0.00, 'scale': 0.03},

            ladd_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            ladd_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            ladd_add={'method': 'normal', 'mean': 0.00, 'scale': 0.03},
            col_rotate={'method': 'normal', 'mean': 0, 'scale': 3})

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
    tmp = zip(net.list_arguments(), shapes[0])
    label_shape = [item for item in tmp if 'label' in item[0]]

    # data loader
    dataiter = dataloader.numpyloader(experiment_name=experiment_name,
                                      dataset=data_set,
                                      augmentation=augment_pipeline,
                                      batch_shape=batch_shape,
                                      label_shape=label_shape,
                                      n_thread=10,
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
            num_epoch= 300)

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


@register
def StereoResnet_flyingthing(epoch, ctx, lr):
    """
      train resnet using flyingthing dataset
    """
    # model name
    experiment_name = 'StereoResnet_flyingthing'

    # shapes
    batch_shape = (4, 3, 384, 768)
    num_iteration = 1600000

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

    net = resnet_symbol.resnet(loss_scale=loss_scale, net_type='stereo', is_sparse=False)

    # dataset
    data_set = dataset.SynthesisData(data_type='stereo',
                                     scene_list=['flyingthing3d'],
                                     rendering_level=['cleanpass'])

    # augmentation setting
    augment_pipeline = augmentation.augmentation(
        interpolation_method='bilinear',
        max_num_tries=10,
        cropped_height=384,
        cropped_width=768,
        data_type='stereo',
        augment_ratio=1.00,
        noise_std=0.03,
        mirror_rate=0.0,
        flip_rate=0.1,
        rotate_range={'method': 'uniform', 'low': 0, 'high': 0},
        translate_range={'method': 'normal', 'mean': 0.0, 'scale': 0.4},
        zoom_range={'method': 'normal', 'mean': 1.3, 'scale': 0.5},
        squeeze_range={'method': 'normal', 'mean': 1.0, 'scale': 0.3},
        gamma_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},
        brightness_range={'method': 'normal', 'mean': 0.0, 'scale': 0.01},
        contrast_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},
        rgb_multiply_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},

        # refer to Dispnet Caffe code
        lmult_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
        lmult_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
        lmult_add={'method': 'normal', 'mean': 0.00, 'scale': 0.02},

        sat_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
        sat_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
        sat_add={'method': 'normal', 'mean': 0.00, 'scale': 0.03},

        col_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
        col_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.2},
        col_add={'method': 'normal', 'mean': 0.00, 'scale': 0.01},

        ladd_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
        ladd_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
        ladd_add={'method': 'normal', 'mean': 0.00, 'scale': 0.03},
        col_rotate={'method': 'normal', 'mean': 0, 'scale': 1})
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
    tmp = zip(net.list_arguments(), shapes[0])
    label_shape = [item for item in tmp if 'label' in item[0]]

    # data loader
    dataiter = dataloader.numpyloader(experiment_name=experiment_name,
                                      dataset=data_set,
                                      augmentation=augment_pipeline,
                                      batch_shape=batch_shape,
                                      label_shape=label_shape,
                                      n_thread=4,
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
            num_epoch=550)
            #int(num_iteration / (dataiter.data_num / batch_shape[0])))

    # save reuslt
    # cannot save CustomOp into json file
    net_saved = resnet_symbol.resnet(loss_scale=loss_scale, net_type='stereo', is_sparse=False)
    args, auxs = mod.get_params()
    model_zoo_path = os.path.join(config.cfg.model.model_zoo, experiment_name)
    mx.model.save_checkpoint(prefix=model_zoo_path,
                             epoch=0,
                             symbol=net_saved,
                             arg_params=args,
                             aux_params=auxs)

    # copy mean to model zoo directory
    shutil.copy2(os.path.join(config.cfg.dataset.mean_dir, experiment_name+'_mean.npy'), config.cfg.model.model_zoo)
    util.generate_deployconfig(experiment_name, 'stereo')

@register
def dispnet_pretrain_rmsprop(epoch, ctx, lr):
    """
      train dispnet using RMSProp
    """
    # model name
    experiment_name = 'dispnet_pretrain_rmsprop'

    # shapes
    batch_shape = (4, 3, 384, 768)
    num_iteration = 1400000

    # optimizer params
    optimizer_type = 'RMSProp'
    optimizer_setting = dict(learning_rate=lr,
                             gamma1=0.9,
                             gamma2=0.9,
                             epsilon=1e-08,
                             wd=0.0004,
                             rescale_grad = 1.0 / batch_shape[0],
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
    augment_pipeline = augmentation.augmentation(
        interpolation_method='bilinear',
        max_num_tries=10,
        cropped_height=384,
        cropped_width=768,
        data_type='stereo',
        augment_ratio=0.8,
        noise_std=0.05,
        mirror_rate=0.0,
        flip_rate=0.1,
        rotate_range={'method': 'uniform', 'low': 0, 'high': 0},
        translate_range={'method': 'normal', 'mean': 0.0, 'scale': 0.4},
        zoom_range={'method': 'normal', 'mean': 1.0, 'scale': 0.2},
        squeeze_range={'method': 'normal', 'mean': 1.0, 'scale': 0.2},
        gamma_range={'method': 'normal', 'mean': 1.0, 'scale': 0.03},
        brightness_range={'method': 'normal', 'mean': 0.0, 'scale': 0.05},
        contrast_range={'method': 'normal', 'mean': 1.0, 'scale': 0.05},
        rgb_multiply_range={'method': 'normal', 'mean': 1.0, 'scale': 0.05},

        lmult_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.5},
        lmult_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.5},
        lmult_add={'method': 'normal', 'mean': 0.00, 'scale': 0.05},

        sat_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.5},
        sat_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.5},
        sat_add={'method': 'normal', 'mean': 0.00, 'scale': 0.05},

        col_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.5},
        col_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.5},
        col_add={'method': 'normal', 'mean': 0.00, 'scale': 0.05},

        ladd_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.5},
        ladd_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.5},
        ladd_add={'method': 'normal', 'mean': 0.00, 'scale': 0.05},
        col_rotate={'method': 'normal', 'mean': 0, 'scale': 5})

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
    tmp = zip(net.list_arguments(), shapes[0])
    label_shape = [item for item in tmp if 'label' in item[0]]

    # training data loader
    dataiter = dataloader.numpyloader(experiment_name=experiment_name,
                                      dataset=data_set,
                                      augmentation=augment_pipeline,
                                      batch_shape=batch_shape,
                                      label_shape=label_shape,
                                      n_thread=10,
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
            num_epoch=200,
            allow_missing=True)
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
    util.generate_deployconfig(experiment_name, 'stereo')

@register
def spynet_flyingchair(epoch, ctx, lr):
    """
      train spynet on flyingchair dataset
    """
    # model name
    experiment_name = 'spynet_flyingchair'
    data_type = 'flow'

    # shapes
    batch_shape = (32, 3, 384, 448)
    num_iteration = 1200000

    # optimizer params
    optimizer_type = 'Adam'
    optimizer_setting = dict(learning_rate=lr,
                             beta1=0.90,
                             beta2=0.999,
                             epsilon=1e-4,
                             rescale_grad=1.0 / batch_shape[0],
                             wd=0.00001,
                             lr_scheduler=mx.lr_scheduler.FactorScheduler(step=250000,
                                                                          factor=0.5,
                                                                          stop_factor_lr=3.125E-6))

    # symbol params
    loss_scale = {'loss1': 0.00,
                  'loss2': 0.00,
                  'loss3': 1.00,
                  'loss4': 1.00,
                  'loss5': 1.00}

    net = spynet_symbol.spynet(loss_scale=loss_scale, net_type=data_type, is_sparse=False)
    fix_params = [item for item in net.list_arguments() if 'upsampling' in item]
    print fix_params
    # dataset
    data_set = dataset.FlyingChairsDataset()
    # augmentation setting
    augment_pipeline = augmentation.augmentation(
        interpolation_method='bilinear',
        max_num_tries=10,
        cropped_height=batch_shape[2],
        cropped_width=batch_shape[3],
        data_type='stereo',
        augment_ratio=0.0,
        noise_std=0.03,
        mirror_rate=0.0,
        flip_rate=0.1,
        rotate_range={'method': 'uniform', 'low': -17, 'high': 17},
        translate_range={'method': 'normal', 'mean': 0.0, 'scale': 0.4},
        zoom_range={'method': 'normal', 'mean': 1.3, 'scale': 0.5},
        squeeze_range={'method': 'normal', 'mean': 1.0, 'scale': 0.3},
        gamma_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},
        brightness_range={'method': 'normal', 'mean': 0.0, 'scale': 0.01},
        contrast_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},
        rgb_multiply_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},

        # refer to Dispnet Caffe code
        lmult_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
        lmult_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
        lmult_add={'method': 'normal', 'mean': 0.00, 'scale': 0.02},

        sat_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
        sat_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
        sat_add={'method': 'normal', 'mean': 0.00, 'scale': 0.03},

        col_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
        col_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.2},
        col_add={'method': 'normal', 'mean': 0.00, 'scale': 0.01},

        ladd_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
        ladd_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
        ladd_add={'method': 'normal', 'mean': 0.00, 'scale': 0.03},
        col_rotate={'method': 'normal', 'mean': 0, 'scale': 1})

    # metric
    eval_metric = [metric.EndPointErr()]

    # initializer
    init = mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)

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
    tmp = zip(net.list_arguments(), shapes[0])
    label_shape = [item for item in tmp if 'label' in item[0]]

    # data loader
    dataiter = dataloader.numpyloader(ctx = ctx,
                                      experiment_name=experiment_name,
                                      dataset=data_set,
                                      augmentation=augment_pipeline,
                                      batch_shape=batch_shape,
                                      label_shape=label_shape,
                                      n_thread=12,
                                      half_life=300000,
                                      initial_coeff=0.00,
                                      final_coeff=1.0,
                                      interpolation_method='bilinear')

    # module
    mod = mx.module.Module(symbol=net,
                           data_names=[item[0] for item in dataiter.provide_data],
                           label_names=[item[0] for item in dataiter.provide_label],
                           context=ctx,
                           fixed_param_names=fix_params)
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
            num_epoch=220,
            allow_missing=True)

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


@register
def flownet2_pretrain(epoch, ctx, lr):
    """
      train flownet on flyingchair dataset
    """
    # model name
    experiment_name = 'flownet2_pretrain'
    data_type = 'flow'

    # shapes
    batch_shape = (4, 3, 384, 768)
    num_iteration = 1200000

    # optimizer params
    optimizer_type = 'Adam'
    optimizer_setting = dict(learning_rate=lr,
                             beta1=0.90,
                             beta2=0.999,
                             epsilon=1e-8,
                             rescale_grad=1.0/batch_shape[0],
                             wd=0.0004,
                             lr_scheduler=mx.lr_scheduler.FactorScheduler(step=250000,
                                                                          factor=0.5,
                                                                          stop_factor_lr=1E-6))

    # symbol params
    loss_scale = {}
    loss_scale['flownetc'] = {'loss1': 0.10,
                              'loss2': 0.00,
                              'loss3': 0.00,
                              'loss4': 0.00,
                              'loss5': 0.00,
                              'loss6': 0.00}


    loss_scale['flownets1'] = {'loss1': 0.10,
                               'loss2': 0.00,
                               'loss3': 0.00,
                               'loss4': 0.00,
                               'loss5': 0.00,
                               'loss6': 0.00}

    loss_scale['flownets2'] = {'loss1': 1.00,
                               'loss2': 0.10,
                               'loss3': 0.00,
                               'loss4': 0.00,
                               'loss5': 0.00,
                               'loss6': 0.00}

    net, flownetc_params = flownet2_symbol.flownet2CSS(loss_scale, net_type='flow', is_sparse=False)

    # fix params
    fix_params = ['upsamplingop_flows1_weight', 'upsamplingop_flows2_weight']
    # fix_params.extend(flownetc_params)

    # dataset
    # data_set = dataset.FlyingChairsDataset()
    # data_set = dataset.MultiDataset(data_type='flow')
    # data_set.register([dataset.SynthesisData(data_type='flow',
    #                                  scene_list=['flyingthing3d'],
    #                                  rendering_level=['cleanpass']),])
    #                    #dataset.FlyingChairsDataset()])

    # augmentation setting
    augment_pipeline = augmentation.augmentation(
        interpolation_method='bilinear',
        max_num_tries=10,
        cropped_height=batch_shape[2],
        cropped_width=batch_shape[3],
        data_type='flow',
        augment_ratio=1.0,
        mirror_rate=0.0,
        flip_rate = 0.0,
        noise_range = {'method':'uniform', 'exp':False, 'mean':0.03, 'spread':0.03},
        translate_range={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.4},
        rotate_range={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.4},
        zoom_range={'method': 'uniform', 'exp': True, 'mean': 0.2, 'spread': 0.4},
        squeeze_range={'method': 'uniform', 'exp': True, 'mean': 0, 'spread': 0.3},

        gamma_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},
        brightness_range={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.02},
        contrast_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},
        rgb_multiply_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},

        lmult_pow={'method': 'uniform', 'exp': True, 'mean': -0.2, 'spread': 0.4},
        lmult_mult={'method': 'uniform', 'exp': True, 'mean': 0.0, 'spread': 0.4},
        lmult_add={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.03},

        sat_pow={'method': 'uniform', 'exp': True, 'mean': 0, 'spread': 0.4},
        sat_mult={'method': 'uniform', 'exp': True, 'mean': -0.3, 'spread': 0.5},
        sat_add={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.03},

        col_pow={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.4},
        col_mult={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.2},
        col_add={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.02},

        ladd_pow={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.4},
        ladd_mult={'method': 'normal', 'exp': True, 'mean': 0.0, 'spread': 0.4},
        ladd_add={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.04},
        col_rotate={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 1})

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

        pretrain_model = os.path.join(config.cfg.model.model_zoo,'flownet_pretrain')
        args, auxs = util.load_checkpoint(pretrain_model, 0)
    else:
        # previous training checkpoint
        args, auxs = util.load_checkpoint(checkpoint_path, epoch)
        # args_new = {}
        # for key in args:
        #     if key in net.list_arguments():
        #         if key.startswith('upsampling'):
        #             continue
        #         args_new[key] = args[key]
        #     else:
        #         print key

        # args = args_new

    # infer shapes of outputs
    shapes = net.infer_shape(img1=batch_shape, img2=batch_shape)
    tmp = zip(net.list_arguments(), shapes[0])
    label_shape = [item for item in tmp if 'label' in item[0]]

    # data loader
    # dataiter = dataloader.numpyloader(ctx=ctx,
    #                                   experiment_name=experiment_name,
    #                                   dataset=data_set,
    #                                   augmentation=augment_pipeline,
    #                                   batch_shape=batch_shape,
    #                                   label_shape=label_shape,
    #                                   n_thread=15,
    #                                   half_life=100000,
    #                                   initial_coeff=0.0,
    #                                   final_coeff=1.0,
    #                                   interpolation_method='bilinear')
    lmdbiter = LMDB.lmdbloader(#lmdb_path='/home/xudong/FlyingChairs_release_lmdb/',
                               lmdb_path='/data/flyingthing_flow_lmdb/',
                               data_type='flow',
                               ctx=ctx,
                               experiment_name=experiment_name,
                               augmentation=augment_pipeline,
                               batch_shape=batch_shape,
                               label_shape=label_shape,
                               interpolation_method='bilinear',
                               use_rnn=False,
                               rnn_hidden_shapes=None,
                               initial_coeff=0.1,
                               final_coeff=1.0,
                               half_life=50000,
                               chunk_size=512,
                               n_thread=10)

    dataiter = mx.io.PrefetchingIter(lmdbiter)

    # module
    mod = mx.module.Module(symbol=net,
                           data_names=[item[0] for item in dataiter.provide_data],
                           label_names=[item[0] for item in dataiter.provide_label],
                           context=ctx,
                           fixed_param_names=fix_params)
    # training
    mod.fit(train_data=dataiter,
            eval_metric=eval_metric,
            epoch_end_callback=mx.callback.module_checkpoint(mod,
                                                             checkpoint_path,
                                                             period=4,
                                                             save_optimizer_states=True),
            batch_end_callback=[mx.callback.Speedometer(batch_shape[0], 20)],
            kvstore='device',
            optimizer=optimizer_type,
            optimizer_params=optimizer_setting,
            initializer=init,
            arg_params=args,
            aux_params=auxs,
            begin_epoch=epoch,
            num_epoch= 430,
            allow_missing=False)

    # save reuslt
    # json cannot save CustomOp
    net_saved, flownetc_params = flownet2_symbol.flownet2CSS(loss_scale, net_type='flow', is_sparse=False)
    args, auxs = mod.get_params()
    model_zoo_path = os.path.join(config.cfg.model.model_zoo, experiment_name)
    mx.model.save_checkpoint(prefix=model_zoo_path,
                             epoch=0,
                             symbol=net_saved,
                             arg_params=args,
                             aux_params=auxs)
    # copy mean file to model zoo directory
    shutil.copy2(os.path.join(config.cfg.dataset.mean_dir, experiment_name + '_mean.npy'), config.cfg.model.model_zoo)
    util.generate_deployconfig(experiment_name, 'flow')

@register
def DRRDispnet(epoch, ctx, lr):
    """
      train dispnet using flyingthing dataset
    """
    # model name
    experiment_name = 'DRRDispnet'

    # shapes
    batch_shape = (16, 3, 320, 768)
    num_iteration = 1400000

    # optimizer params
    optimizer_setting = dict(learning_rate=lr,
                             beta1=0.90,
                             beta2=0.999,
                             epsilon=1e-4,
                             wd=0.0004,
                             rescale_grad = 1.0,
                             lr_scheduler = mx.lr_scheduler.FactorScheduler(step=100000,
                                                                          factor=0.5,
                                                                          stop_factor_lr=1E-6))
    optimizer = mx.optimizer.Adam(**optimizer_setting)

    # symbol params
    loss_scale = {'loss0': 1.00,
                  'loss1': 0.10,
                  'loss2': 0.00,
                  'loss3': 0.00,
                  'loss4': 0.00,
                  'loss5': 0.00,
                  'loss6': 0.00}

    net = DRR_symbol.DRR_Dispnet(loss_scale=loss_scale, net_type='stereo', is_sparse=True)
    # lr_mult = {item : 1.0 for item in net.list_arguments() if 'DRR' not in item}
    # optimizer.set_lr_mult(lr_mult)
    fix_params = ['upsample_pr1_weight']
    # dataset
    # data_set = dataset.MultiDataset(data_type='stereo')
    # dataset_list = [dataset.KittiDataset('stereo', '2015', is_train=True) for i in range(200)]
    # dataset_list.append(dataset.SynthesisData(data_type='stereo',
    #                                  scene_list=['flyingthing3d'],
    #                                  rendering_level=['cleanpass']))
    # data_set.register(dataset_list)

    # augmentation setting
    augment_pipeline = augmentation.augmentation(
        interpolation_method='bilinear',
        max_num_tries=10,
        cropped_height=batch_shape[2],
        cropped_width=batch_shape[3],
        data_type='stereo',
        augment_ratio=1.0,
        mirror_rate=0.0,
        flip_rate = 0.0,
        noise_range = {'method':'uniform', 'exp':False, 'mean':0.03, 'spread':0.03},
        translate_range={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.4},
        rotate_range={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.4},
        zoom_range={'method': 'uniform', 'exp': True, 'mean': 0.2, 'spread': 0.4},
        squeeze_range={'method': 'uniform', 'exp': True, 'mean': 0, 'spread': 0.3},

        gamma_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},
        brightness_range={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.02},
        contrast_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},
        rgb_multiply_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},

        lmult_pow={'method': 'uniform', 'exp': True, 'mean': -0.2, 'spread': 0.4},
        lmult_mult={'method': 'uniform', 'exp': True, 'mean': 0.0, 'spread': 0.4},
        lmult_add={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.03},

        sat_pow={'method': 'uniform', 'exp': True, 'mean': 0, 'spread': 0.4},
        sat_mult={'method': 'uniform', 'exp': True, 'mean': -0.3, 'spread': 0.5},
        sat_add={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.03},

        col_pow={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.4},
        col_mult={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.2},
        col_add={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.02},

        ladd_pow={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.4},
        ladd_mult={'method': 'normal', 'exp': True, 'mean': 0.0, 'spread': 0.4},
        ladd_add={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.04},
        col_rotate={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 1})

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
        pretrain_model = os.path.join(config.cfg.model.model_zoo, 'dispnet_pretrain')
        args, auxs = util.load_checkpoint(pretrain_model, 0)
    else:
        # previous training checkpoint
        args, auxs = util.load_checkpoint(checkpoint_path, epoch)

    # infer shapes of outputs
    shapes = net.infer_shape(img1=batch_shape, img2=batch_shape)
    tmp = zip(net.list_arguments(), shapes[0])
    label_shape = [item for item in tmp if 'label' in item[0]]

    # training data loader
    # dataiter = dataloader.numpyloader(experiment_name=experiment_name,
    #                                   dataset=data_set,
    #                                   augmentation=augment_pipeline,
    #                                   batch_shape=batch_shape,
    #                                   label_shape=label_shape,
    #                                   n_thread=10,
    #                                   half_life=10000,
    #                                   initial_coeff=0.0,
    #                                   final_coeff=1.0,
    #                                   interpolation_method='nearest')

    lmdbiter = LMDB.lmdbloader(lmdb_path='/home/xudong/FlyingThings3D_release_TRAIN_lmdb/',
                                data_type='stereo',
                                ctx=ctx,
                                experiment_name=experiment_name,
                                augmentation=augment_pipeline,
                                batch_shape=batch_shape,
                                label_shape=label_shape,
                                interpolation_method='bilinear',
                                use_rnn=False,
                                rnn_hidden_shapes=None,
                                initial_coeff=0.1,
                                final_coeff=1.0,
                                half_life=50000,
                                chunk_size=32,
                                n_thread=20)

    dataiter = mx.io.PrefetchingIter(lmdbiter)

    # module
    mod = mx.module.Module(symbol=net,
                           data_names=[item[0] for item in dataiter.provide_data],
                           label_names=[item[0] for item in dataiter.provide_label],
                           context=ctx,
                           fixed_param_names=fix_params)
    # training
    mod.fit(train_data=dataiter,
            eval_metric=eval_metric,
            epoch_end_callback=mx.callback.module_checkpoint(mod,
                                                             checkpoint_path,
                                                             period=2,
                                                             save_optimizer_states=True),
            batch_end_callback=[mx.callback.Speedometer(batch_shape[0], 20)],
            kvstore='device',
            optimizer=optimizer,
            initializer=init,
            arg_params=args,
            aux_params=auxs,
            begin_epoch=epoch,
            num_epoch=210,
            allow_missing=True)
            #int(num_iteration / (dataiter.data_num / batch_shape[0])))

    # save reuslt
    # cannot save CustomOp into json file
    net_saved = DRR_symbol.DRR_Dispnet(loss_scale=loss_scale, net_type='stereo', is_sparse=False)
    args, auxs = mod.get_params()
    model_zoo_path = os.path.join(config.cfg.model.model_zoo, experiment_name)
    mx.model.save_checkpoint(prefix=model_zoo_path,
                             epoch=0,
                             symbol=net_saved,
                             arg_params=args,
                             aux_params=auxs)

    # copy mean to model zoo directory
    shutil.copy2(os.path.join(config.cfg.dataset.mean_dir, experiment_name+'_mean.npy'), config.cfg.model.model_zoo)
    util.generate_deployconfig(experiment_name, 'stereo')


@register
def flownethalf(epoch, ctx, lr):

    # model name
    experiment_name = 'flownethalf'
    data_type = 'flow'

    # shapes
    batch_shape = (4, 3, 384, 512)
    num_iteration = 1200000

    # optimizer params
    optimizer_type = 'Adam'
    optimizer_setting = dict(learning_rate=lr,
                             beta1=0.90,
                             beta2=0.999,
                             epsilon=1e-4,
                             rescale_grad=1.0/batch_shape[0],
                             wd=0.0004,
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

    net = flownets_half_symbol.flownets_half(loss_scale=loss_scale, net_type=data_type, is_sparse=False)

    # dataset
    data_set = dataset.FlyingChairsDataset()
    # data_set = dataset.SynthesisData(data_type='flow',
    #                                  scene_list=['flyingthing3d'],
    #                                  rendering_level=['cleanpass'])
    # augmentation setting
    augment_pipeline = augmentation.augmentation(
            interpolation_method='bilinear',
            max_num_tries=10,
            cropped_height=384,
            cropped_width=512,
            data_type='stereo',
            augment_ratio=1.0,
            noise_std=0.03,
            mirror_rate=0.0,
            flip_rate = 0.1,
            rotate_range={'method': 'uniform', 'low': -17, 'high': 17},
            translate_range={'method': 'normal', 'mean': 0.0, 'scale': 0.4},
            zoom_range={'method': 'normal', 'mean': 1.0, 'scale': 0.3},
            squeeze_range={'method': 'normal', 'mean': 1.0, 'scale': 0.3},
            gamma_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},
            brightness_range={'method': 'normal', 'mean': 0.0, 'scale': 0.01},
            contrast_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},
            rgb_multiply_range={'method': 'normal', 'mean': 1.0, 'scale': 0.01},

            # refer to Dispnet Caffe code
            lmult_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            lmult_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            lmult_add={'method': 'normal', 'mean': 0.00, 'scale': 0.03},

            sat_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            sat_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            sat_add={'method': 'normal', 'mean': 0.00, 'scale': 0.03},

            col_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            col_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            col_add={'method': 'normal', 'mean': 0.00, 'scale': 0.03},

            ladd_pow={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            ladd_mult={'method': 'normal', 'mean': 1.00, 'scale': 0.3},
            ladd_add={'method': 'normal', 'mean': 0.00, 'scale': 0.03},
            col_rotate={'method': 'normal', 'mean': 0, 'scale': 3})

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
    tmp = zip(net.list_arguments(), shapes[0])
    label_shape = [item for item in tmp if 'label' in item[0]]

    # data loader
    dataiter = dataloader.numpyloader(experiment_name=experiment_name,
                                      dataset=data_set,
                                      augmentation=augment_pipeline,
                                      batch_shape=batch_shape,
                                      label_shape=label_shape,
                                      n_thread=6,
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
            num_epoch= 180)

    # save reuslt
    # json cannot save CustomOp
    net_saved = flownets_half_symbol.flownets_half(loss_scale=loss_scale, net_type=data_type, is_sparse=False)
    args, auxs = mod.get_params()
    model_zoo_path = os.path.join(config.cfg.model.model_zoo, experiment_name)
    mx.model.save_checkpoint(prefix=model_zoo_path,
                             epoch=0,
                             symbol=net_saved,
                             arg_params=args,
                             aux_params=auxs)

    # copy mean file to model zoo directory
    shutil.copy2(os.path.join(config.cfg.dataset.mean_dir, experiment_name + '_mean.npy'), config.cfg.model.model_zoo)
    util.generate_deployconfig(experiment_name, 'flow')


@register
def flownet2ss(epoch, ctx, lr):

    # model name
    experiment_name = 'flownet2ss'
    data_type = 'flow'

    # shapes
    batch_shape = (8, 3, 384, 448)
    num_iteration = 1200000

    # optimizer params
    optimizer_type = 'Adam'
    optimizer_setting = dict(learning_rate=lr,
                             beta1=0.90,
                             beta2=0.999,
                             epsilon=1e-4,
                             rescale_grad=1.0/batch_shape[0],
                             wd=0.0004,
                             lr_scheduler=mx.lr_scheduler.FactorScheduler(step=250000,
                                                                          factor=0.5,
                                                                          stop_factor_lr=1E-6))

    # symbol params
    loss_scale = {}
    loss_scale['flownets1'] = {'loss1': 1.00,
                               'loss2': 0.00,
                               'loss3': 0.00,
                               'loss4': 0.00,
                               'loss5': 0.00,
                               'loss6': 0.00}

    loss_scale['flownets2'] = {'loss1': 1.00,
                               'loss2': 0.00,
                               'loss3': 0.00,
                               'loss4': 0.00,
                               'loss5': 0.00,
                               'loss6': 0.00}

    net = flownet2ss_symbol.flownet2ss(loss_scale, net_type='flow', is_sparse=False)

    # fix params
    fix_params = ['upsamplingop_flows1_weight']

    # dataset
    data_set = dataset.FlyingChairsDataset()
    # data_set = dataset.MultiDataset(data_type='flow')
    # data_set.register([dataset.SynthesisData(data_type='flow',
    #                                          scene_list=['flyingthing3d'],
    #                                          rendering_level=['cleanpass']),])
                       #dataset.FlyingChairsDataset()])

    # augmentation setting
    augment_pipeline = augmentation.augmentation(
        interpolation_method='bilinear',
        max_num_tries=10,
        cropped_height=batch_shape[2],
        cropped_width=batch_shape[3],
        data_type='flow',
        augment_ratio=0.7,
        mirror_rate=0.0,
        flip_rate = 0.0,
        noise_range = {'method':'uniform', 'exp':False, 'mean':0.03, 'spread':0.03},
        translate_range={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.4},
        rotate_range={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.4},
        zoom_range={'method': 'uniform', 'exp': True, 'mean': 0.2, 'spread': 0.4},
        squeeze_range={'method': 'uniform', 'exp': True, 'mean': 0, 'spread': 0.3},

        gamma_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},
        brightness_range={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.02},
        contrast_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},
        rgb_multiply_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},

        lmult_pow={'method': 'uniform', 'exp': True, 'mean': -0.2, 'spread': 0.4},
        lmult_mult={'method': 'uniform', 'exp': True, 'mean': 0.0, 'spread': 0.4},
        lmult_add={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.03},

        sat_pow={'method': 'uniform', 'exp': True, 'mean': 0, 'spread': 0.4},
        sat_mult={'method': 'uniform', 'exp': True, 'mean': -0.3, 'spread': 0.5},
        sat_add={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.03},

        col_pow={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.4},
        col_mult={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.2},
        col_add={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.02},

        ladd_pow={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.4},
        ladd_mult={'method': 'normal', 'exp': True, 'mean': 0.0, 'spread': 0.4},
        ladd_add={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.04},
        col_rotate={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 1})

    # metric
    eval_metric = [metric.EndPointErr()]

    # initializer
    init = mx.initializer.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)

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
    tmp = zip(net.list_arguments(), shapes[0])
    label_shape = [item for item in tmp if 'label' in item[0]]

    # data loader
    # dataiter = dataloader.numpyloader(ctx=ctx,
    #                                   experiment_name=experiment_name,
    #                                   dataset=data_set,
    #                                   augmentation=augment_pipeline,
    #                                   batch_shape=batch_shape,
    #                                   label_shape=label_shape,
    #                                   n_thread=15,
    #                                   half_life=100000,
    #                                   initial_coeff=0.0,
    #                                   final_coeff=1.0,
    #                                   interpolation_method='bilinear')
    lmdbiter = LMDB.lmdbloader(#lmdb_path='/home/xudong/FlyingChairs_release_lmdb/',
                               lmdb_path='/data/flyingthing_flow_lmdb/',
                               data_type='flow',
                               ctx=ctx,
                               experiment_name=experiment_name,
                               augmentation=augment_pipeline,
                               batch_shape=batch_shape,
                               label_shape=label_shape,
                               interpolation_method='bilinear',
                               use_rnn=False,
                               rnn_hidden_shapes=None,
                               initial_coeff=0.1,
                               final_coeff=1.0,
                               half_life=50000,
                               chunk_size=32,
                               n_thread=8)

    dataiter = mx.io.PrefetchingIter(lmdbiter)

    # module
    mod = mx.module.Module(symbol=net,
                           data_names=[item[0] for item in dataiter.provide_data],
                           label_names=[item[0] for item in dataiter.provide_label],
                           context=ctx,
                           fixed_param_names=fix_params)
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
            num_epoch= 500,
            allow_missing=True)

    # save reuslt
    # json cannot save CustomOp
    net_saved = flownet2ss_symbol.flownet2ss(loss_scale, net_type='flow', is_sparse=False)
    args, auxs = mod.get_params()
    model_zoo_path = os.path.join(config.cfg.model.model_zoo, experiment_name)
    mx.model.save_checkpoint(prefix=model_zoo_path,
                             epoch=0,
                             symbol=net_saved,
                             arg_params=args,
                             aux_params=auxs)
    # copy mean file to model zoo directory
    shutil.copy2(os.path.join(config.cfg.dataset.mean_dir, experiment_name + '_mean.npy'), config.cfg.model.model_zoo)
    util.generate_deployconfig(experiment_name, 'flow')

@register
def dispnetCSS_pretrain(epoch, ctx, lr):
    """
      train dispnetCSS on flyingthing
    """
    # model name
    experiment_name = 'dispnetCSS_pretrain'
    data_type = 'stereo'

    # shapes
    batch_shape = (4, 3, 320, 768)
    num_iteration = 1200000

    # optimizer params
    optimizer_type = 'Adam'
    optimizer_setting = dict(learning_rate=lr,
                             beta1=0.90,
                             beta2=0.999,
                             epsilon=1e-8,
                             rescale_grad=1.0/batch_shape[0],
                             wd=0.0004,
                             lr_scheduler=mx.lr_scheduler.FactorScheduler(step=250000,
                                                                          factor=0.5,
                                                                          stop_factor_lr=1E-6))

    # symbol params
    loss_scale = {}
    loss_scale['dispnetc'] = {'loss1': 1.00,
                              'loss2': 0.00,
                              'loss3': 0.00,
                              'loss4': 0.00,
                              'loss5': 0.00,
                              'loss6': 0.00}


    loss_scale['dispnets1'] = {'loss1': 1.00,
                               'loss2': 0.00,
                               'loss3': 0.00,
                               'loss4': 0.00,
                               'loss5': 0.00,
                               'loss6': 0.00}

    loss_scale['dispnets2'] = {'loss1': 1.00,
                               'loss2': 0.00,
                               'loss3': 0.00,
                               'loss4': 0.00,
                               'loss5': 0.00,
                               'loss6': 0.00}

    net, flownetc_params = dispnet2CSS_symbol.dispnet2CSS(loss_scale, net_type=data_type, is_sparse=True)

    # fix params
    fix_params = ['upsamplingop_disps1_weight', 'upsamplingop_disps2_weight']
    # fix_params.extend(flownetc_params)

    # dataset
    # data_set = dataset.FlyingChairsDataset()
    data_set = dataset.MultiDataset(data_type=data_type)
    kitti = dataset.KittiDataset(data_type, '2015', is_train=True)
    kitti.dirs = kitti.dirs[:160]
    data_set.register([dataset.SynthesisData(data_type=data_type,
                                             scene_list=['Driving', 'flyingthing3d', 'Monkaa'],
                                             rendering_level=['cleanpass']),
                       kitti])
    #dataset.FlyingChairsDataset()])

    # augmentation setting
    augment_pipeline = augmentation.augmentation(
        interpolation_method='bilinear',
        max_num_tries=10,
        cropped_height=batch_shape[2],
        cropped_width=batch_shape[3],
        data_type=data_type,
        augment_ratio=1.0,
        mirror_rate=0.0,
        flip_rate = 0.0,
        noise_range = {'method':'uniform', 'exp':False, 'mean':0.03, 'spread':0.03},
        translate_range={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.4},
        rotate_range={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.0},
        zoom_range={'method': 'uniform', 'exp': True, 'mean': 0.2, 'spread': 0.4},
        squeeze_range={'method': 'uniform', 'exp': True, 'mean': 0, 'spread': 0.3},

        gamma_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},
        brightness_range={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.02},
        contrast_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},
        rgb_multiply_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},

        lmult_pow={'method': 'uniform', 'exp': True, 'mean': -0.2, 'spread': 0.4},
        lmult_mult={'method': 'uniform', 'exp': True, 'mean': 0.0, 'spread': 0.4},
        lmult_add={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.03},

        sat_pow={'method': 'uniform', 'exp': True, 'mean': 0, 'spread': 0.4},
        sat_mult={'method': 'uniform', 'exp': True, 'mean': -0.3, 'spread': 0.5},
        sat_add={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.03},

        col_pow={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.4},
        col_mult={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.2},
        col_add={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.02},

        ladd_pow={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.4},
        ladd_mult={'method': 'normal', 'exp': True, 'mean': 0.0, 'spread': 0.4},
        ladd_add={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.04},
        col_rotate={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 1})

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
        pretrain_model = os.path.join(config.cfg.model.model_zoo, 'dispnet_pretrain')
        args, auxs = util.load_checkpoint(pretrain_model, 0)
    else:
        # previous training checkpoint
        args, auxs = util.load_checkpoint(checkpoint_path, epoch)

    # infer shapes of outputs
    shapes = net.infer_shape(img1=batch_shape, img2=batch_shape)
    tmp = zip(net.list_arguments(), shapes[0])
    label_shape = [item for item in tmp if 'label' in item[0]]

    # data loader
    dataiter = dataloader.numpyloader(ctx=ctx,
                                      experiment_name=experiment_name,
                                      dataset=data_set,
                                      augmentation=augment_pipeline,
                                      batch_shape=batch_shape,
                                      label_shape=label_shape,
                                      n_thread=20,
                                      half_life=100000,
                                      initial_coeff=0.0,
                                      final_coeff=1.0,
                                      interpolation_method='bilinear')
    # dataiter = LMDB.lmdbloader(#lmdb_path='/home/xudong/FlyingChairs_release_lmdb/',
    #     #lmdb_path='/data/flyingthing_flow_lmdb/',
    #     lmdb_path='/home/xudong/FlyingThings3D_release_TRAIN_lmdb/',
    #     data_type=data_type,
    #     ctx=ctx,
    #     experiment_name=experiment_name,
    #     augmentation=augment_pipeline,
    #     batch_shape=batch_shape,
    #     label_shape=label_shape,
    #     interpolation_method='bilinear',
    #     use_rnn=False,
    #     rnn_hidden_shapes=None,
    #     initial_coeff=0.0,
    #     final_coeff=1.0,
    #     half_life=50000,
    #     chunk_size=3000,
    #     n_thread=20)

    dataiter = mx.io.PrefetchingIter(dataiter)

    # module
    mod = mx.module.Module(symbol=net,
                           data_names=[item[0] for item in dataiter.provide_data],
                           label_names=[item[0] for item in dataiter.provide_label],
                           context=ctx,
                           fixed_param_names=fix_params)
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
            num_epoch= 300,
            allow_missing=True)

    # save reuslt
    # json cannot save CustomOp
    net_saved, dispnetc_params = dispnet2CSS_symbol.dispnet2CSS(loss_scale, net_type=data_type, is_sparse=False)
    args, auxs = mod.get_params()
    model_zoo_path = os.path.join(config.cfg.model.model_zoo, experiment_name)
    mx.model.save_checkpoint(prefix=model_zoo_path,
                             epoch=0,
                             symbol=net_saved,
                             arg_params=args,
                             aux_params=auxs)
    # copy mean file to model zoo directory
    shutil.copy2(os.path.join(config.cfg.dataset.mean_dir, experiment_name + '_mean.npy'), config.cfg.model.model_zoo)
    util.generate_deployconfig(experiment_name, data_type)


@register
def flownet2CSS_origin(epoch, ctx, lr):
    """
      train flownet on flyingchair dataset
    """

    # model name
    experiment_name = 'flownet2CSS_origin'
    data_type = 'flow'

    # shapes
    batch_shape = (8, 3, 320, 448)
    num_iteration = 1200000

    # optimizer params
    optimizer_type = 'Adam'
    optimizer_setting = dict(learning_rate=lr,
                             beta1=0.90,
                             beta2=0.999,
                             epsilon=1e-8,
                             rescale_grad=1.0/batch_shape[0],
                             wd=0.0004,
                             lr_scheduler=mx.lr_scheduler.FactorScheduler(step=250000,
                                                                          factor=0.5,
                                                                          stop_factor_lr=1E-6))

    # symbol params
    loss_scale = {}
    loss_scale['flownetc'] = {'loss2': 1.00,
                              'loss3': 0.16,
                              'loss4': 0.04,
                              'loss5': 0.02,
                              'loss6': 0.01}


    loss_scale['flownets1'] = {'loss2': 0.005,
                               'loss3': 0.01,
                               'loss4': 0.02,
                               'loss5': 0.08,
                               'loss6': 0.32}

    loss_scale['flownets2'] = {'loss2': 0.005,
                               'loss3': 0.01,
                               'loss4': 0.02,
                               'loss5': 0.08,
                               'loss6': 0.32}

    net, flownetc_params = flownet2CSS_origin_symbol.flownet2CSS(loss_scale, net_type='flow', is_sparse=False)

    # fix params
    # fix_params = ['upsamplingop_flows1_weight', 'upsamplingop_flows2_weight']
    fix_params = None
    # fix_params.extend(flownetc_params)

    # dataset
    # data_set = dataset.FlyingChairsDataset()
    data_set = dataset.MultiDataset(data_type='flow')
    data_set.register([dataset.SynthesisData(data_type='flow',
                                             scene_list=['flyingthing3d'],
                                             rendering_level=['cleanpass']),])
    #dataset.FlyingChairsDataset()])

    # augmentation setting
    augment_pipeline = augmentation.augmentation(
        interpolation_method='bilinear',
        max_num_tries=10,
        cropped_height=batch_shape[2],
        cropped_width=batch_shape[3],
        data_type='flow',
        augment_ratio=1.0,
        mirror_rate=0.0,
        flip_rate = 0.0,
        noise_range = {'method':'uniform', 'exp':False, 'mean':0.03, 'spread':0.03},
        translate_range={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.4},
        rotate_range={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.4},
        zoom_range={'method': 'uniform', 'exp': True, 'mean': 0.2, 'spread': 0.4},
        squeeze_range={'method': 'uniform', 'exp': True, 'mean': 0, 'spread': 0.3},

        gamma_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},
        brightness_range={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.02},
        contrast_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},
        rgb_multiply_range={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.02},

        lmult_pow={'method': 'uniform', 'exp': True, 'mean': -0.2, 'spread': 0.4},
        lmult_mult={'method': 'uniform', 'exp': True, 'mean': 0.0, 'spread': 0.4},
        lmult_add={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.03},

        sat_pow={'method': 'uniform', 'exp': True, 'mean': 0, 'spread': 0.4},
        sat_mult={'method': 'uniform', 'exp': True, 'mean': -0.3, 'spread': 0.5},
        sat_add={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 0.03},

        col_pow={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.4},
        col_mult={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.2},
        col_add={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.02},

        ladd_pow={'method': 'normal', 'exp': True, 'mean': 0, 'spread': 0.4},
        ladd_mult={'method': 'normal', 'exp': True, 'mean': 0.0, 'spread': 0.4},
        ladd_add={'method': 'normal', 'exp': False, 'mean': 0, 'spread': 0.04},
        col_rotate={'method': 'uniform', 'exp': False, 'mean': 0, 'spread': 1})

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

        # pretrain_model = os.path.join(config.cfg.model.model_zoo,'flownet_pretrain')
        # args, auxs = util.load_checkpoint(pretrain_model, 0)
        args = None
        auxs = None
    else:
        # previous training checkpoint
        args, auxs = util.load_checkpoint(checkpoint_path, epoch)
        # args_new = {}
        # for key in args:
        #     if key in net.list_arguments():
        #         if key.startswith('upsampling'):
        #             continue
        #         args_new[key] = args[key]
        #     else:
        #         print key

        # args = args_new

    # infer shapes of outputs
    shapes = net.infer_shape(img1=batch_shape, img2=batch_shape)
    tmp = zip(net.list_arguments(), shapes[0])
    label_shape = [item for item in tmp if 'label' in item[0]]

    # data loader
    # dataiter = dataloader.numpyloader(ctx=ctx,
    #                                   experiment_name=experiment_name,
    #                                   dataset=data_set,
    #                                   augmentation=augment_pipeline,
    #                                   batch_shape=batch_shape,
    #                                   label_shape=label_shape,
    #                                   n_thread=15,
    #                                   half_life=100000,
    #                                   initial_coeff=0.0,
    #                                   final_coeff=1.0,
    #                                   interpolation_method='bilinear')
    lmdbiter = LMDB.lmdbloader(#lmdb_path='/home/xudong/FlyingChairs_release_lmdb/',
        lmdb_path='/data/flyingthing_flow_lmdb/',
        data_type='flow',
        ctx=ctx,
        experiment_name=experiment_name,
        augmentation=augment_pipeline,
        batch_shape=batch_shape,
        label_shape=label_shape,
        interpolation_method='bilinear',
        use_rnn=False,
        rnn_hidden_shapes=None,
        initial_coeff=0.1,
        final_coeff=1.0,
        half_life=50000,
        chunk_size=4096,
        n_thread=25)

    dataiter = mx.io.PrefetchingIter(lmdbiter)

    # module
    mod = mx.module.Module(symbol=net,
                           data_names=[item[0] for item in dataiter.provide_data],
                           label_names=[item[0] for item in dataiter.provide_label],
                           context=ctx,
                           fixed_param_names=fix_params)
    # training
    mod.fit(train_data=dataiter,
            eval_metric=eval_metric,
            epoch_end_callback=mx.callback.module_checkpoint(mod,
                                                             checkpoint_path,
                                                             period=1,
                                                             save_optimizer_states=True),
            batch_end_callback=[mx.callback.Speedometer(batch_shape[0], 20)],
            kvstore='device',
            optimizer=optimizer_type,
            optimizer_params=optimizer_setting,
            initializer=init,
            arg_params=args,
            aux_params=auxs,
            begin_epoch=epoch,
            num_epoch= 530,
            allow_missing=True)

    # save reuslt
    # json cannot save CustomOp
    net_saved, flownetc_params = flownet2CSS_origin_symbol.flownet2CSS(loss_scale, net_type='flow', is_sparse=False)
    args, auxs = mod.get_params()
    model_zoo_path = os.path.join(config.cfg.model.model_zoo, experiment_name)
    mx.model.save_checkpoint(prefix=model_zoo_path,
                             epoch=0,
                             symbol=net_saved,
                             arg_params=args,
                             aux_params=auxs)
    # copy mean file to model zoo directory
    shutil.copy2(os.path.join(config.cfg.dataset.mean_dir, experiment_name + '_mean.npy'), config.cfg.model.model_zoo)
    util.generate_deployconfig(experiment_name, 'flow')