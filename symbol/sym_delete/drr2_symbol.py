import mxnet as mx
from symbol.enet_symbol import  get_conv

from symbol.sym_delete.res_unit import residual_unit


def detect(data, output_dim, name):

    data = get_conv(name='detect.0'+name, data=data, num_filter=32, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9)
    data = mx.sym.Pooling(data=data,kernel=(2, 2),pool_type='max',stride=(2,2),name='detect.pool.0'+name)
    tmp2 = data = get_conv(name='detect.1'+name, data=data, num_filter=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9)
    data = mx.sym.Pooling(data=data, kernel=(2, 2), pool_type='max', stride=(2, 2), name='detect.pool.1'+name)
    data = get_conv(name='detect.2'+name, data=data, num_filter=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9)
    data = get_conv(name='detect.3'+name, data=data, num_filter=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9)
    data = get_conv(name='detect.4'+name, data=data, num_filter=64, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + tmp2
    data = get_conv(name='detect.5'+name, data=data, num_filter=32, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = get_conv(name='detect.6'+name, data=data, num_filter=output_dim, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                    with_relu=False, bn_momentum=0.9, is_conv=True)
    data = mx.sym.Activation(data=data,act_type='sigmoid',name='predict_error_map'+name)

    return data

def replace(data, output_dim, name):

    encoder0 = data = residual_unit(data=data, num_filter=64, stride=(2, 2), dim_match=False, name='replace.0'+name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    encoder1 = data = residual_unit(data=data, num_filter=128, stride=(2, 2), dim_match=False, name='replace.1'+name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    encoder2 = data = residual_unit(data=data, num_filter=256, stride=(2, 2), dim_match=False, name='replace.2'+name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    encoder3 = data = residual_unit(data=data, num_filter=512, stride=(2, 2), dim_match=False, name='replace.3'+name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    encoder4 = data = residual_unit(data=data, num_filter=512, stride=(2, 2), dim_match=False, name='replace.4'+name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    data = residual_unit(data=data, num_filter=1024, stride=(2, 2), dim_match=False, name='replace.5'+name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    data = get_conv(name='replace.6'+name, data=data, num_filter=512, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder4
    data = get_conv(name='replace.7'+name, data=data, num_filter=512, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder3
    data = get_conv(name='replace.8'+name, data=data, num_filter=256, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder2
    data = get_conv(name='replace.9'+name, data=data, num_filter=128, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder1
    data = get_conv(name='replace.10'+name, data=data, num_filter=64, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder0
    data = get_conv(name='replace.11'+name, data=data, num_filter=output_dim, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    return data

def refine(data, output_dim, name):

    data = residual_unit(data=data, num_filter=64, stride=(2, 2), dim_match=False, name='refine.0.0' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    encoder0 = data = residual_unit(data=data, num_filter=64, stride=(1, 1), dim_match=False, name='refine.0.1' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)

    data = residual_unit(data=data, num_filter=128, stride=(2, 2), dim_match=False, name='refine.1.0' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    encoder1 = data = residual_unit(data=data, num_filter=128, stride=(1, 1), dim_match=False, name='refine.1.1' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)

    data = residual_unit(data=data, num_filter=256, stride=(2, 2), dim_match=False, name='refine.2.0' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    encoder2 = data = residual_unit(data=data, num_filter=256, stride=(1, 1), dim_match=False, name='refine.2.1' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)

    data = residual_unit(data=data, num_filter=512, stride=(2, 2), dim_match=False, name='refine.3.0' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)
    data = residual_unit(data=data, num_filter=512, stride=(1, 1), dim_match=False, name='refine.3.1' + name,
                         bottle_neck=False, bn_mom=0.9, memonger=False)

    data = get_conv(name='refine.decoder0' + name, data=data, num_filter=256, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
             with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder2
    data = get_conv(name='refine.decoder1' + name, data=data, num_filter=128, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder1
    data = get_conv(name='refine.decoder2' + name, data=data, num_filter=64, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
                    with_relu=True, bn_momentum=0.9, is_conv=False)
    data = data + encoder0
    data = get_conv(name='refine.decoder3' + name, data=data, num_filter=output_dim, kernel=(4, 4), stride=(2, 2), pad=(1, 1),
             with_relu=False, bn_momentum=0.9, is_conv=False)
    return data


def detect_replace_refine(init_label, input_feature, output_dim, name):

    # detect error map
    data = mx.sym.Concat(init_label, input_feature)
    map = detect(data, output_dim, name)

    # replace
    data = mx.sym.Concat(data, map)
    replace_value = replace(data, output_dim, name)
    U = map * replace_value + (1-map) * init_label

    # refine
    data = mx.sym.Concat(data, U)
    refine_value = refine(data, output_dim, name)
    U = U + refine_value

    return U, map, replace_value, refine_value






