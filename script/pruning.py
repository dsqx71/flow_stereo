
# coding: utf-8

# In[52]:

import mxnet as mx
import json
import os
import copy
import cv2
import time
import logging
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logging.debug("test")
from flow_stereo.symbol import dispnet_symbol
from flow_stereo.data import dataset, dataloader, LMDB, augmentation, config
from flow_stereo.others import util,visualize, metric

get_ipython().magic(u'matplotlib inline')


# In[53]:

import mxnet as mx
import numpy as np
from numpy.linalg import lstsq

def rank_paramer(net):
    """
    Get the number of parameter in each layer
    """
    shapes = zip(net.list_arguments(), net.infer_shape(img1=batch_shape, img2=batch_shape)[0])
    params_num = []
    for item in shapes:
        tot = 1
        for i in item[1]:
            tot *=i
        params_num.append([item[0], tot])

#     params_num.sort(cmp=lambda x,y: cmp(x[1],y[1]), reverse=True)
    return params_num

def modify_symbol(sym, layer_name, which_attr, value):
    """
    modify attribute of the specified layer
    """
    conf = json.loads(sym.tojson())
    for index in range(len(conf["nodes"])):
        if  layer_name == conf["nodes"][index]["name"] or             layer_name + "_weight" == conf["nodes"][index]["name"] or             layer_name + "_bias" == conf["nodes"][index]["name"]:
            conf["nodes"][index]["attr"][which_attr] = value
    
    if layer_name.startswith('conv1') or layer_name.startswith('conv2'):
        layer_name = layer_name[:5] + 'img2'
        for index in range(len(conf["nodes"])):
            if  layer_name == conf["nodes"][index]["name"] or                 layer_name + "_weight" == conf["nodes"][index]["name"] or                 layer_name + "_bias" == conf["nodes"][index]["name"]:
                conf["nodes"][index]["attr"][which_attr] = value

    sym = mx.sym.load_json(json.dumps(conf))
    return sym  
    
def get_topo(net):
    """
    Get network topo
    """
    conf = json.loads(net.tojson())
    nodes = conf['nodes']
    net_topo = {}
    for node in nodes:
        name = node["name"]
        net_topo[name] = {'prev':[], 'next':[]}
    for node in nodes:
        name = node["name"]
        inputs = node["inputs"]
        net_topo[name]['type'] = node["op"]
        if "attr" in node:
            net_topo[name]["attr"] = node["attr"]
        for item in inputs:
            input_node = nodes[item[0]]
            input_name = input_node["name"]
            net_topo[name]['prev'].append(input_name)
            net_topo[input_name]['next'].append(name)
    return net_topo

def replace_weight(name, op_type, arg_params, retain_index):
    # convert to numpy for slicing
    weight = arg_params['%s_weight' % name].asnumpy() # [out, in, k, k]
    if op_type == 'Convolution':
        arg_params['%s_weight' % name] = mx.nd.array(weight[retain_index])
    elif op_type == 'Deconvolution':
        arg_params['%s_weight' % name] = mx.nd.array(weight[:, retain_index])
    if '%s_bias' % name in arg_params:
        bias = arg_params['%s_bias' % name].asnumpy() # [out]
        arg_params['%s_bias' % name] = mx.nd.array(bias[retain_index])

def get_quality(ac, q_type='variance'):
    print('quality type: %s' % q_type)
    if q_type == 'variance':
        if ac.ndim == 4: 
            return ac.mean(3).mean(2).transpose().var(axis=1)
        if ac.ndim == 2: 
            return ac.transpose().var(axis=1)
        
def get_retainIndex(quality, retain_num):
    
    retain_index = quality.argmax()
    retain_index = np.reshape(retain_index, (1,)) # for hstack
    quality[retain_index] = 0
    while retain_index.shape[0] < retain_num:
        new_index = quality.argmax()
        quality[new_index] = 0
        retain_index = np.hstack((retain_index, new_index))
    return retain_index


def rewrite(ac, layer_retain_index, layer_prune_index, channels, retain_num, num_samples,
            height, width, arg_params, next_layer, op_type):
    v_sample = ac[:, layer_retain_index]
#     print v_sample.shape
    v_sample = np.transpose(v_sample, (0, 2, 3, 1))
#     print num_samples, height, width, retain_num
    v_sample = np.reshape(v_sample, (num_samples * height * width, retain_num), order = 'F')
    v_nosample = ac[:, layer_prune_index]
    v_nosample = np.transpose(v_nosample, (0, 2, 3, 1))
    v_nosample = np.reshape(v_nosample, (num_samples * height * width, channels - retain_num), order = 'F')
    print v_sample.shape, v_nosample.shape
    alpha = lstsq(v_sample, v_nosample)[0] # [retain_num, prune_num] prune_num = channels - retain_num
    print 'lst done'
    
    if next_layer == 'conv2img1':
        next_layer = 'share2'
        
    next_weight = arg_params['%s_weight' % next_layer].asnumpy()
    next_weight_outc, next_weight_inc, next_weight_h, next_weight_w = next_weight.shape
    
    if op_type == "Convolution":
        next_weight_retain = next_weight[:, layer_retain_index] # [out, retain_num, k, k]
        next_weight_retain = np.transpose(next_weight_retain, (1, 0, 2, 3)) # [retain_num, out, k, k]
        next_weight_retain = np.reshape(next_weight_retain,
                                        (retain_num, next_weight_outc * next_weight_h * next_weight_w), order = 'F') # [retain_num, out * k * k]

        next_weight_prune = next_weight[:, layer_prune_index] # [out, prune_num, k, k]
        next_weight_prune = np.transpose(next_weight_prune, (1, 0, 2, 3)) # [prune_num, out, k, k]
        next_weight_prune = np.reshape(next_weight_prune,
                                       (next_weight_inc - retain_num, next_weight_outc * next_weight_h * next_weight_w), order = 'F') # [prune_num, out * k * k]

        # [retain_num, out * k * k] + [retain_num, prune_num] * [prune_num, out * k * k]
        next_weight_retain_rw = next_weight_retain + np.dot(alpha, next_weight_prune)
        next_weight_retain_rw = np.reshape(next_weight_retain_rw,
                                           (retain_num, next_weight_outc, next_weight_h, next_weight_w), order = 'F') # [retain_num, out, k, k]
        next_weight_retain_rw = np.transpose(next_weight_retain_rw, (1, 0, 2, 3)) # [out, retain_num, k, k]
    elif op_type  == 'Deconvolution':
        next_weight_retain = next_weight[layer_retain_index]
        next_weight_retain = np.reshape(next_weight_retain,
                                        (retain_num, next_weight_inc * next_weight_h * next_weight_w), order = 'F')

        next_weight_prune = next_weight[layer_prune_index]
        next_weight_prune = np.reshape(next_weight_prune,
                                       (next_weight_outc - retain_num, next_weight_inc * next_weight_h * next_weight_w), order = 'F') # [prune_num, out * k * k]

        next_weight_retain_rw = next_weight_retain + np.dot(alpha, next_weight_prune)
        next_weight_retain_rw = np.reshape(next_weight_retain_rw,
                                           (retain_num, next_weight_inc, next_weight_h, next_weight_w), order = 'F') # [retain_num, out, k, k]
    
    arg_params['%s_weight' % next_layer] = mx.nd.array(next_weight_retain_rw)
    
def prune_vgg(ac, layer, act_name, op_type, next_layers, arg_params, retain_num, 
              net_topo, concat_act=None, quality_type = 'variance'):
    
    arg_params = arg_params.copy()
    if ac.ndim == 4:
        print('pruning conv layer...')
        num_samples, channels, height, width = ac.shape
        prune_num = channels - retain_num
        quality = get_quality(ac, q_type=quality_type)
        retain_index = get_retainIndex(quality, retain_num)
        layer_retain_index = np.ones(channels)>1
        layer_retain_index[retain_index] = True
        layer_prune_index = np.logical_not(layer_retain_index)
        if layer=='conv1img1' or layer=='conv2img1':
            layer_tmp = 'share%s' % layer[4]
        else:
            layer_tmp = layer
        replace_weight(layer_tmp, op_type, arg_params, layer_retain_index)
        index = 0
        for next_layer in next_layers:
            next_layer, is_concat, op_type = next_layer
            if 'correlation' in next_layer:
                continue
            if is_concat == 'concat':
                ac_concat =  concat_act[index]
                num_samples = ac_concat.shape[0]
                layer_retain_index_concat = []
                for i in range(len(topo[next_layer]['prev'])):
                    if topo[next_layer]['prev'][i] == act_name:
                        concat_layer_index = layer_retain_index
                    else:
                        try:
                            channel_concat = topo[topo[topo[next_layer]['prev'][i]]['prev'][0]]['attr']['num_filter']
                        except:
                            channel_concat = 81
                        concat_layer_index = np.ones(int(channel_concat)).astype(bool)
                    layer_retain_index_concat.append(concat_layer_index)
                index += 1
                layer_retain_index_concat = np.hstack(layer_retain_index_concat)
                channels_concat = len(layer_retain_index_concat)
                retain_num_concat = channels_concat - prune_num
                layer_prune_index_concat = ~layer_retain_index_concat
                height_concat, width_concat = ac_concat.shape[2:]
                rewrite(ac=ac_concat, 
                        layer_retain_index=layer_retain_index_concat, 
                        layer_prune_index=layer_prune_index_concat, 
                        channels=channels_concat, 
                        retain_num=retain_num_concat,
                        num_samples=num_samples,
                        height=height_concat, 
                        width=width_concat,
                        arg_params=arg_params, 
                        next_layer=topo[next_layer]['next'][0],
                        op_type=topo[topo[next_layer]['next'][0]]['type'])
            else:
                rewrite(ac, layer_retain_index, layer_prune_index, channels, 
                        retain_num, num_samples,height, width, arg_params, next_layer, op_type)
                
    print 'done'
    return arg_params


# In[54]:

# setting
experiment_name = 'dispnet_finetune'
data_type = 'stereo'
batch_shape = (4, 3, 320, 768)
test_shape = (1, 3, 384, 1280)
interpolation = 'nearest'
epoch = 2650
ctx = [mx.gpu(4), mx.gpu(5), mx.gpu(6), mx.gpu(7)]
loss_scale = {'loss1': 1.00,
              'loss2': 0.00,
              'loss3': 0.00,
              'loss4': 0.00,
              'loss5': 0.00,
              'loss6': 0.00}

# parameter
checkpoint_prefix = os.path.join(config.cfg.model.check_point, experiment_name)
checkpoint_path = os.path.join(checkpoint_prefix, experiment_name)    
checkpoint_path = os.path.join(checkpoint_prefix, experiment_name)
args, auxs = util.load_checkpoint(checkpoint_path, epoch)

# symbol
net = dispnet_symbol.dispnet(loss_scale=loss_scale, net_type='stereo', is_sparse=False)
tmp_net = dispnet_symbol.dispnet(loss_scale=loss_scale, net_type='stereo', is_sparse=True)
net = net.get_internals()

# infer label shape
shapes = net.infer_shape(img1=batch_shape, img2=batch_shape)
tmp = zip(net.list_arguments(), shapes[0])
label_shape = [item for item in tmp if 'label' in item[0]]

# dataset 
data_set = dataset.KittiDataset('stereo', '2015', is_train=True)
# data_set.dirs.extend(data_set.dirs)
# data_set.dirs.extend(data_set.dirs)
# data_set.dirs.extend(data_set.dirs)

# augmentation
augment_pipeline = augmentation.augmentation(
        interpolation_method=interpolation,
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
    
# dataiter 
dataiter = dataloader.numpyloader(ctx=ctx,
                                  experiment_name=experiment_name,
                                  dataset=data_set,
                                  augmentation=augment_pipeline,
                                  batch_shape=batch_shape,
                                  label_shape=label_shape,
                                  n_thread=2,
                                  half_life=100000,
                                  initial_coeff=0.0,
                                  final_coeff=1.0,
                                  interpolation_method=interpolation)

# build module
mod = mx.module.Module(symbol=net,
                       data_names=[item[0] for item in dataiter.provide_data],
                       label_names=[item[0] for item in dataiter.provide_label],
                       context=ctx,
                       fixed_param_names=None)
mod.bind(data_shapes=dataiter.provide_data, label_shapes=dataiter.provide_label, for_training=True, force_rebind=False)
mod.set_params(arg_params=args, aux_params=auxs, allow_missing=False)


# In[55]:

param_rank = rank_paramer(net)
topo = get_topo(net)
pruing_queue = []
for item in param_rank:
    
    name = item[0]
    
    # not prune siamese network
#     if 'share' in name and :
#         continue
        
    # must have params
    if 'weight' not in name:
        continue
    
    # Get layer name
    layer_name = topo[name]['next'][0]
    activation_name = topo[layer_name]['next'][0]
    
    if 'share' not in layer_name and 'relu' not in activation_name:
        continue
        
    next_layer = []
    for item in topo[activation_name]['next']:
        if not 'concat' in  item:
            next_layer.append((item, 'not_concat'))
        else:
            next_layer.append((item, 'concat'))
    
    pruing_queue.append({'activation_name':activation_name, 
                         'layer_name': layer_name,
                         'next_layer': next_layer })


# In[56]:

pruing_queue[-2:]


# In[ ]:




# In[57]:

for item in pruing_queue[-2:]:
    print item['layer_name']
    
    activation_list = {}
    activation_list[item['activation_name']] = [] 
    for next_layer in item['next_layer']:
        if next_layer[1] == 'concat':
            activation_list[next_layer[0]] = []
    num_filter = int(topo[item['layer_name']]['attr']['num_filter'])
    retain_num = int(num_filter * 0.8) 
    dataiter.reset()
    # Get activation        
    for batch in dataiter:
        mod.forward(batch, is_train=False)
        arg = dict(zip(mod.output_names, mod.get_outputs()))
        for i in activation_list.keys():
            name = i + '_output'
            activation_list[i].append(arg[name].asnumpy())
        if item['layer_name'].startswith('conv1') or item['layer_name'].startswith('conv2'):
            siamese_name = item['layer_name'][:5] + 'img2'
            siamese_activation = topo[siamese_name]['next'][0] + '_output'
            activation_list[item['activation_name']].append(arg[siamese_activation].asnumpy())
    # Concat samples
    for key in activation_list:
        activation_list[key] = np.concatenate(activation_list[key], axis=0)
    
    new_args = prune_vgg(ac=activation_list[item['activation_name']], 
                         act_name = item['activation_name'],
                         layer=item['layer_name'], 
                         op_type = topo[item['layer_name']]['type'],
                         next_layers=[j + (topo[j[0]]['type'],) for j in item['next_layer']],
                         concat_act=[activation_list[j[0]] for j in item['next_layer'] if topo[j[0]]['type'] == 'Concat'],
                         arg_params=args, 
                         net_topo = topo,
                         retain_num=retain_num)

    new_net = modify_symbol(sym=net, layer_name=item['layer_name'], which_attr="num_filter", value=str(retain_num))
    tmp_net = modify_symbol(sym=tmp_net, layer_name=item['layer_name'], which_attr="num_filter", value=str(retain_num))
    net = new_net
    args = new_args
    topo = get_topo(net)
#     rebind
    mod = mx.module.Module(symbol=tmp_net,
                           data_names=[item[0] for item in dataiter.provide_data],
                           label_names=[item[0] for item in dataiter.provide_label],
                           context=ctx,
                           fixed_param_names=None)
    mod.bind(data_shapes=dataiter.provide_data, label_shapes=dataiter.provide_label, for_training=True, force_rebind=False)
    mod.set_params(arg_params=args, aux_params=auxs, allow_missing=False)
    
    eval_metric = [metric.D1all(), metric.EndPointErr()]
    optimizer_type = 'Adam'
    optimizer_setting = dict(learning_rate = 1e-5,
                             beta1 = 0.90,
                             beta2 = 0.999,
                             epsilon = 1e-4,
                             wd = 0.0000)
    dataiter.reset()
    print 'fine tune....'
    mod.fit(train_data=dataiter,
            eval_metric=eval_metric,  
            batch_end_callback=[mx.callback.Speedometer(batch_shape[0], 100)],
            kvstore='device',
            optimizer=optimizer_type,
            optimizer_params=optimizer_setting,
            begin_epoch=0,
            num_epoch=40)
    print 'fine tune done'
    args, auxs = mod.get_params()
    mod = mx.module.Module(symbol=net,
                           data_names=[item[0] for item in dataiter.provide_data],
                           label_names=[item[0] for item in dataiter.provide_label],
                           context=ctx,
                           fixed_param_names=None)
    mod.bind(data_shapes=dataiter.provide_data, label_shapes=dataiter.provide_label, for_training=True, force_rebind=False)
    mod.set_params(arg_params=args, aux_params=auxs, allow_missing=False)
    
    
    # test
    model = mx.model.FeedForward(ctx=mx.gpu(4),
                                 symbol=new_net,
                                 arg_params=new_args,
                                 aux_params=auxs,
                                 numpy_batch_size=1)
    
    test_set = dataset.KittiDataset(data_type='stereo', which_year='2012', is_train=True)
    error = 0.0
    count = 0.0
    for dirs in test_set.dirs:
        img1, img2, label, _ = test_set.get_data(dirs)
        img1 = img1 * 0.00392156862745098
        img2 = img2 * 0.00392156862745098

        img1 = img1 - img1.reshape(-1,3).mean(0)
        img2 = img2 - img2.reshape(-1,3).mean(0)

        img1 = cv2.resize(img1,(test_shape[3], test_shape[2]))
        img2 = cv2.resize(img2,(test_shape[3], test_shape[2]))

        img1 = np.expand_dims(img1.transpose(2, 0, 1), 0)
        img2 = np.expand_dims(img2.transpose(2, 0, 1), 0)

        batch = mx.io.NDArrayIter(data = {'img1':img1, 'img2':img2})
        pred = model.predict(batch)[-1]
        ret = cv2.resize(pred[0,0], label.shape[::-1]) * label.shape[1] / float(test_shape[-1])

        err = np.power(ret-label, 2)
        err = np.power(err, 0.5)
        error += err[err==err].sum()
        count += (err==err).sum()

    #     visualize.plot(ret, 'prediction', waitforkey=False)
    #     visualize.plot(err, 'error',waitforkey=False)
    print 'total EPE : ', error/count


# In[58]:

exe = net.simple_bind(img1=test_shape, img2=test_shape, ctx=mx.gpu(0))


# In[59]:

for key in args:
    if key in exe.arg_dict:
        exe.arg_dict[key][:] = args[key]


# In[62]:

tot = 0.0
tot2 = 0.0
for i in range(100):
    tic = time.time()

    exe.arg_dict['img1'][:] = img1
    exe.arg_dict['img2'][:] = img2
    tot2 += time.time() - tic
    exe.forward()
    exe.outputs[-1].asnumpy()
    toc = time.time()
    tot += toc - tic
tot /= 100
tot2 /=100
print tot, tot2


# In[63]:

old_args, auxs = util.load_checkpoint(checkpoint_path, epoch)


# In[64]:

prefix = '/data/checkpoint_flowstereo/dispnet_pruning/dispnet_finetune_pruning_all_ft_80%'
mx.model.save_checkpoint(prefix, 0, net,args, auxs)


# In[ ]:



