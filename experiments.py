"""
In machine learing experiments, "Hard code" is inevitable.  You should carefully review the code below and
corresponding symbol.

The design purpose of experiments.py is to isolate "hard code" of experiments from reset components, and help
users strictly perform experiments.
"""
import mxnet as mx
import os
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
def dispnet_finetune(epoch, ctx):
    """
    Load caffe pretrain model, and fine tune dispnet using KITTI 2015 stereo dataset
    """

    # model name
    experiment_name = 'dispnet_finetune'
    pretrain_model = 'caffe_pretrain'

    # shapes
    batch_shape = (4, 3, 320, 768)
    num_iteration = 80000

    # optimizer params
    optimizer_type = 'Adam'
    optimizer_setting = dict(learning_rate = 1e-5,
                             beta1 = 0.90,
                             beta2 = 0.999,
                             epsilon = 1e-4,
                             decay_factor = 0.0004)
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
                                                             checkpoint_prefix+'dispnet_finetune',
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

