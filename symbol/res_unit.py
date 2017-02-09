import mxnet as mx

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False,
                  factor=0.25):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    # if bottle_neck:
    #     # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
    #     bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    #     act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    #     conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*factor), kernel=(1,1), stride=(1,1), pad=(0,0),
    #                                   no_bias=True, workspace=workspace, name=name + '_conv1')
    #     bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    #     act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    #     conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*factor), kernel=(3,3), stride=stride, pad=(1,1),
    #                                   no_bias=True, workspace=workspace, name=name + '_conv2')
    #     bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    #     act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    #     conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
    #                                workspace=workspace, name=name + '_conv3')
    #     if dim_match:
    #         shortcut = data
    #     else:
    #         shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
    #                                         workspace=workspace, name=name+'_sc')
    #     if memonger:
    #         shortcut._set_attr(mirror_stage='True')
    #     return conv3 + shortcut
    # else:
    conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                               no_bias=False, workspace=workspace, name=name + '_conv1')
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=1e-5 + 1e-10, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')

    conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=False, workspace=workspace, name=name + '_conv2')
    bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=1e-5 + 1e-10, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')

    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=False,
                                        workspace=workspace, name=name+'_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    return act2 + shortcut