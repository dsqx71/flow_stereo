import mxnet as mx
from dispnet_symbol import get_loss
from collections import namedtuple
from res_unit import residual_unit
from enet_symbol import get_conv

LSTMState = namedtuple('LSTMState', ['h', 'c'])
LSTMParam = namedtuple('LSTMParam', ['i2h_weight', 'i2h_bias',
                                     'h2h_weight', 'h2h_bias'])

def lstm(num_hidden, in_data, prev_state, param, seqidx, layeridx=0, dropout=0.):

    """LSTM Cell Symbol"""
    if dropout > 0.:
        in_data = mx.sym.Dropout(data=in_data, p=dropout)
    i2h = mx.sym.FullyConnected(data=in_data,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=4*num_hidden,
                                name="t%d_l%d_i2h1" % (seqidx, layeridx))

    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=4*num_hidden,
                                name="t%d_l%d_h2h1" % (seqidx, layeridx))

    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(data=slice_gates[0], act_type='sigmoid')
    in_transform = mx.sym.Activation(data=slice_gates[1], act_type='tanh')
    forget_gate = mx.sym.Activation(data=slice_gates[2], act_type='sigmoid')
    out_gate = mx.sym.Activation(data=slice_gates[3], act_type='sigmoid')

    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type='tanh')
    return LSTMState(c=next_c, h=next_h)


def lstm_unroll(seq_len, in_data, num_hidden, param, init_state, get_all_hiddens=False):

    hidden_all = []
    prev_state1 = init_state[0]
    prev_state2 = init_state[1]
    prev_state3 = init_state[2]
    prev_state4 = init_state[3]

    for seqidx in range(seq_len):
        x = in_data[seqidx]
        prev_state1 = lstm(num_hidden=num_hidden, in_data=x, prev_state=prev_state1,
                          param=param[0], seqidx=seqidx)
        prev_state2 = lstm(num_hidden=num_hidden, in_data=prev_state1.h+x, prev_state=prev_state2,
                           param=param[1], seqidx=seqidx)
        prev_state3 = lstm(num_hidden=num_hidden, in_data=prev_state1.h+prev_state2.h+x, prev_state=prev_state3,
                           param=param[2], seqidx=seqidx)
        prev_state4 = lstm(num_hidden=num_hidden, in_data=prev_state3.h+prev_state2.h+x, prev_state=prev_state4,
                           param=param[3], seqidx=seqidx)
        hidden = prev_state4.h + prev_state3.h + prev_state2.h + prev_state1.h
        hidden_all.append(hidden)

    return hidden_all if get_all_hiddens else [prev_state1, prev_state2, prev_state3, prev_state4]


def init_lstm_param(name):

    return LSTMParam(i2h_weight=mx.sym.Variable('%s_i2h_weight' % name),
                     i2h_bias=mx.sym.Variable('%s_i2h_bias' % name),
                     h2h_weight=mx.sym.Variable('%s_h2h_weight' % name),
                     h2h_bias=mx.sym.Variable('%s_h2h_bias' % name))


def stereo_rnn(img1, img2, height, width, num_hidden, name, init_state):

    encoder_param = [init_lstm_param('encoder{}'.format(i) + name) for i in range(4)]

    decoder_param = [init_lstm_param('decoder{}'.format(i) + name) for i in range(4)]

    img1 = mx.sym.Reshape(img1, shape=(0, 0, -1))
    img2 = mx.sym.Reshape(img2, shape=(0, 0, -1))
    img1_features = mx.sym.SliceChannel(img1, num_outputs=height, axis=2,
                                        name='img1_slice' + name)
    img2_features = mx.sym.SliceChannel(img2, num_outputs=height, axis=2,
                                        name='img2_slice' + name)

    hidden_all = []
    init_state = [init_state for i in range(4)]
    for i in range(height):
        encoder_in_data = mx.sym.SliceChannel(img1_features[i],
                                              num_outputs=width, axis=2,
                                              squeeze_axis=True,
                                              name='img1_row%i_slice' + name)
        decoder_in_data = mx.sym.SliceChannel(img2_features[i],
                                              num_outputs=width, axis=2,
                                              squeeze_axis=True,
                                              name='img2_row%i_slice' + name)
        encoder_state = lstm_unroll(width, encoder_in_data, num_hidden=num_hidden,
                                    param=encoder_param, init_state=init_state,
                                    get_all_hiddens=False)
        hiddens = lstm_unroll(width, decoder_in_data, num_hidden=num_hidden,
                              param=decoder_param, init_state=encoder_state,
                              get_all_hiddens=True)
        hidden_all += hiddens
    hidden_all = [mx.sym.expand_dims(s, axis=2) for s in hidden_all]
    hidden_all = mx.sym.Concat(*hidden_all, dim=2)
    hidden_all = mx.sym.Reshape(hidden_all, shape=(0, 0, height, width))
    hidden_all = mx.sym.Convolution(hidden_all, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=num_hidden, name='pr' + name)
    return hidden_all



def rnn(height,width,num_hidden,bn_momentum=0.9):

    data = {}
    weight = {}
    bias = {}
    init_state = LSTMState(c=mx.sym.Variable('init_c'),
                           h=mx.sym.Variable('init_h'))
    img1 = data[0] = mx.sym.Variable('img1')
    img2 = data[1] = mx.sym.Variable('img2')
    label = mx.sym.Variable('stereo_downsample1')
    data[2] = mx.sym.Pooling(data=data[0], kernel=(2,2),  stride=(2, 2), pool_type='avg')
    data[3] = mx.sym.Pooling(data=data[1], kernel=(2,2),  stride=(2, 2), pool_type='avg')

    for num_layer in range(1, 5):

        weight[0] = mx.sym.Variable('share_l%d_blue_weight' % num_layer)
        bias[0] = mx.sym.Variable('share_%d_blue_bias' % num_layer)

        weight[1] = mx.sym.Variable('share_l%d_red_weight' % num_layer)
        bias[1] = mx.sym.Variable('share_%d_red_bias' % num_layer)

        if num_layer <= 2:
            kernel = (3, 3)
            pad = (1,1)
            num_filter = 32
        else:
            kernel = (5, 5)
            pad = (2,2)
            num_filter = 200

        stride=(2,2) if num_layer==1 else (1,1)
        for j in range(4):

            data[j] = mx.sym.Convolution(data=data[j], weight=weight[j / 2], bias=bias[j / 2], kernel=kernel,
                                         num_filter=num_filter, pad=pad,stride=stride)
            data[j] = mx.sym.Activation(data=data[j], act_type='relu')


    for j in range(4):
        data[j] = residual_unit(data=data[j], num_filter=num_hidden, stride=(1, 1), dim_match=False, name='block_redir{}'.format(j),
                         bottle_neck=False, bn_mom=bn_momentum, workspace=512, memonger=False)

    pr1 = stereo_rnn(data[1], data[0], height / 2, width / 2, num_hidden, 'original', init_state)
    pr2 = stereo_rnn(data[3], data[2], height / 4, width / 4, num_hidden, 'downsample', init_state)
    pr2 = mx.sym.Deconvolution(pr2, pad=(1, 1), kernel=(4, 4), stride=(2, 2), num_filter=1, name='upconv', no_bias=False)

    data = mx.sym.Concat(pr1, pr2, data[1],data[0])
    data = residual_unit(data=data, num_filter=64, stride=(1, 1), dim_match=False, name='block1',
                               bottle_neck=False, bn_mom=bn_momentum, workspace=512, memonger=False)
    data = residual_unit(data=data, num_filter=32, stride=(1, 1), dim_match=False, name='block2',
                         bottle_neck=False, bn_mom=bn_momentum, workspace=512, memonger=False)
    data = residual_unit(data=data, num_filter=16, stride=(1, 1), dim_match=False, name='block3',
                         bottle_neck=False, bn_mom=bn_momentum, workspace=512, memonger=False)

    pr = mx.symbol.Convolution(name='predict', data=data, num_filter=1, kernel=(3, 3), stride=(1, 1),
                               pad=(1, 1), dilate=(1, 1), no_bias=False)
    net = get_loss(pr, label, 1.0, name='loss')
    return net

def stereo_mxnet_rnn(img1, img2, height, width, num_hidden, name, init_state):

    encoder_param = mx.sym.Variable(name+"encoder_bias")
    decoder_param = mx.sym.Variable(name+"decoder_bias")

    init_c = mx.sym.SwapAxis(init_state.c, dim1=0, dim2=1)
    init_h = mx.sym.SwapAxis(init_state.h, dim1=0, dim2=1)

    # conv features should be processed by BN
    img1 = mx.sym.Reshape(img1, shape=(0, 0, -1))
    img2 = mx.sym.Reshape(img2, shape=(0, 0, -1))
    img1_features = mx.sym.SliceChannel(img1, num_outputs=height, axis=2,
                                        name='img1_slice'+name)
    img2_features = mx.sym.SliceChannel(img2, num_outputs=height, axis=2,
                                        name='img2_slice'+name)

    hidden_all = []
    tmps = []
    for i in range(height):
        encoder_in_data = img1_features[i]
        encoder_in_data = mx.sym.transpose(encoder_in_data, axes=(2, 0, 1))
        decoder_in_data = img2_features[i]
        decoder_in_data = mx.sym.transpose(decoder_in_data, axes=(2, 0, 1))

        encoder_state = mx.sym.RNN(data=encoder_in_data, state_size=num_hidden,
                                   num_layers=1, mode='lstm',
                                   name='encoder'+name,
                                   state=init_h,
                                   state_cell=init_c,
                                   parameters=encoder_param,
                                   state_outputs=True)
        # tmp = mx.sym.BlockGrad(data=encoder_state[0],name='tmp{}'.format(i)+name)
        # tmps.append(tmp)
        hiddens = mx.sym.RNN(data=decoder_in_data, state_size=num_hidden,
                             num_layers=1, mode='lstm',
                             name='decoder'+name,
                             state=encoder_state['encoder{}_state'.format(name)],
                             state_cell=encoder_state['encoder{}_state_cell'.format(name)],
                             parameters=decoder_param)
        hidden_all.append(hiddens)
    hidden_all = [mx.sym.transpose(s, axes=(1, 2, 0)) for s in hidden_all]
    hidden_all = mx.sym.Concat(*hidden_all, dim=2)
    hidden_all = mx.sym.Reshape(hidden_all, shape=(0, 0, height, width))
    # tmp = data = residual_unit(data=hidden_all, num_filter=128, stride=(1, 1), dim_match=False, name='block1'+name,
    #               bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False)
    # data = residual_unit(data=data, num_filter=128, stride=(1, 1), dim_match=False, name='block2' + name,
    #                      bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False)
    # data = residual_unit(data=data, num_filter=128, stride=(1, 1), dim_match=False, name='block3' + name,
    #                      bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False)
    # data = data + tmp
    #
    # pr = mx.sym.Convolution(data, pad=(1, 1), kernel=(3, 3), stride=(1, 1), num_filter=1, name='pr' + name)
    return hidden_all
