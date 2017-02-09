from collections import namedtuple

import mxnet as mx

from symbol.enet_symbol import get_conv

RNNState = namedtuple('RNNState', ['h'])
RNNParam = namedtuple('RNNParam', ['i2h_weight', 'i2h_bias',
                                   'h2h_weight', 'h2h_bias',
                                   'h2o_weight', 'h2o_bias'])

GRUState = namedtuple('GRUState', ['h'])
GRUParam = namedtuple('GRUParam', ['i2g_weight', 'i2g_bias',
                                   'h2g_weight', 'h2g_bias',
                                   'i2h_weight', 'i2h_bias',
                                   'h2h_weight', 'h2h_bias',
                                   'h2o_weight', 'h2o_bias'])

def rnn(num_hidden, in_data, prev_hidden, param, seqidx, seqidy, direct):

    i2h = mx.sym.FullyConnected(data=in_data,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden,
                                name='{}_y{}_x{}_i2h'.format(direct, seqidy, seqidx)
                                )
    h2h = mx.sym.FullyConnected(data=prev_hidden,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden,
                                name='{}_y{}_x{}_h2h'.format(direct, seqidy, seqidx)
                                )
    hidden = i2h + h2h
    hidden = mx.sym.Activation(data=hidden, act_type='relu')

    return RNNState(h=hidden)


def gru(num_hidden, in_data, prev_hidden, param, seqidx, seqidy, direct):

    i2g = mx.sym.FullyConnected(data=in_data,
                               weight=param.i2g_weight,
                               bias=param.i2g_bias,
                               num_hidden=2*num_hidden,
                               name='{}_y{}_x{}_i2g'.format(direct, seqidy, seqidx))
    h2g = mx.sym.FullyConnected(data=prev_hidden,
                               weight=param.h2g_weight,
                               bias=param.h2g_bias,
                               num_hidden=2*num_hidden,
                               name='{}_y{}_x{}_h2g'.format(direct, seqidy, seqidx))
    gates = mx.sym.Activation(i2g + h2g, act_type="sigmoid")
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=2,
                                      name='{}_y{}_x{}_slice'.format(direct, seqidy, seqidx))
    # 2 gates
    z = slice_gates[0]
    r = slice_gates[1]

    rxh = r * prev_hidden
    i2h = mx.sym.FullyConnected(data=in_data,
                               weight=param.i2h_weight,
                               bias=param.i2h_bias,
                               num_hidden=num_hidden,
                               name='{}_y{}_x{}_i2g'.format(direct, seqidy, seqidx))
    h2h = mx.sym.FullyConnected(data=rxh,
                               weight=param.h2h_weight,
                               bias=param.h2h_bias,
                               num_hidden=num_hidden,
                               name='{}_y{}_x{}_h2g'.format(direct, seqidy, seqidx))
    h_ = mx.sym.Activation(i2h + h2h, act_type="tanh")
    hidden = (1 - z) * prev_hidden + z * h_

    return GRUState(h=hidden)


def init_hw_matrix(height, width):
    """
    Initialize a matrix of size height x width, filled with None
    """
    return [[None for _ in xrange(width)] for _ in xrange(height)]


def init_rnn_param(name):

    param = RNNParam(i2h_weight=mx.symbol.Variable('%s_rnn_i2h_weight' % name),
                     i2h_bias=mx.symbol.Variable('%s_rnn_i2h_bias' % name),
                     h2h_weight=mx.symbol.Variable('%s_rnn_h2h_weight' % name),
                     h2h_bias=mx.symbol.Variable('%s_rnn_h2h_bias' % name),
                     h2o_weight=mx.symbol.Variable('%s_rnn_h2o_weight' % name),
                     h2o_bias=mx.symbol.Variable('%s_rnn_h2o_bias' % name))
    return param


def init_gru_param(name):
    param = GRUParam(i2g_weight=mx.symbol.Variable('%s_gru_i2g_weight' % name),
                     i2g_bias=mx.symbol.Variable('%s_gru_i2g_bias' % name),
                     i2h_weight=mx.symbol.Variable('%s_gru_i2h_weight' % name),
                     i2h_bias=mx.symbol.Variable('%s_gru_i2h_bias' % name),
                     h2g_weight=mx.symbol.Variable('%s_gru_h2g_weight' % name),
                     h2g_bias=mx.symbol.Variable('%s_gru_h2g_bias' % name),
                     h2h_weight=mx.symbol.Variable('%s_gru_h2h_weight' % name),
                     h2h_bias=mx.symbol.Variable('%s_gru_h2h_bias' % name),
                     h2o_weight=mx.symbol.Variable('%s_gru_h2o_weight' % name),
                     h2o_bias=mx.symbol.Variable('%s_gru_h2o_bias' % name))
    return param


def rnn_southeast(local_features, height, width, num_hidden, init_state, use_gru=False):
    """ SouthEast Direction
    Args:
        local_features: list of tensors
        height: int
        width: int
    """
    # params should share
    hiddens = init_hw_matrix(height, width)
    if use_gru:
        rnn_f = gru
        param = init_gru_param('se')
    else:
        rnn_f = rnn
        param = init_rnn_param('se')

    for h in xrange(height):
        for w in xrange(width):
            loc_feat = local_features[h * width + w]
            if w == 0 and h == 0:
                prev_hidden = init_state.h
            elif w > 0 and h == 0:
                assert hiddens[h][w-1] is not None, "Hidden state is not initialized yet."
                prev_hidden = hiddens[h][w-1]
            elif w == 0 and h > 0:
                assert hiddens[h-1][w] is not None, "Hidden state is not initialized yet."
                prev_hidden = hiddens[h-1][w]
            else:
                assert hiddens[h-1][w] is not None and hiddens[h][w-1] is not None, \
                        "Hidden state is not initialized yet."
                prev_hidden = hiddens[h-1][w] + hiddens[h][w-1]

            next_state = rnn_f(num_hidden, loc_feat, prev_hidden, param, w, h, 'se')
            hiddens[h][w] = next_state.h

    outputs = reduce(lambda x, y: x + y, hiddens) # list of symbols, [(b, c), ..]
    outputs = [mx.symbol.expand_dims(data=s, axis=2) for s in outputs]
    outputs = mx.symbol.Concat(*outputs, dim=2)
    outputs = mx.symbol.Reshape(data=outputs, shape=(0, 0, height, width))
    outputs = get_conv(data=outputs, num_filter=num_hidden, kernel=(1, 1),
                       stride=(1, 1), pad=(0, 0), with_relu=False,
                       name='se_projection')
    return outputs


def rnn_southwest(local_features, height, width, num_hidden, init_state, use_gru=False):
    """ SouthWest Direction
    """
    hiddens = init_hw_matrix(height, width)
    if use_gru:
        rnn_f = gru
        param = init_gru_param('sw')
    else:
        rnn_f = rnn
        param = init_rnn_param('sw')

    for h in xrange(height):
        for w in reversed(xrange(width)):
            loc_feat = local_features[h * width + w]
            if w == width - 1 and h == 0:
                prev_hidden = init_state.h
            elif w < width - 1 and h == 0:
                assert hiddens[h][w+1] is not None, "Hidden state is not initialized yet."
                prev_hidden = hiddens[h][w+1]
            elif w == width - 1 and h > 0:
                assert hiddens[h-1][w] is not None, "Hidden state is not initialized yet."
                prev_hidden = hiddens[h-1][w]
            else:
                assert hiddens[h-1][w] is not None and hiddens[h][w:1] is not None, \
                        "Hidden state is not initialized yet."
                prev_hidden = hiddens[h-1][w] + hiddens[h][w+1]

            next_state = rnn_f(num_hidden, loc_feat, prev_hidden, param, w, h, 'sw')
            hiddens[h][w] = next_state.h
    outputs = reduce(lambda x, y: x + y, hiddens) # list of symbols, [(b, c), ..]
    outputs = [mx.symbol.expand_dims(data=s, axis=2) for s in outputs]
    outputs = mx.symbol.Concat(*outputs, dim=2)
    outputs = mx.symbol.Reshape(data=outputs, shape=(0, 0, height, width))
    outputs = get_conv(data=outputs, num_filter=num_hidden, kernel=(1, 1),
                       stride=(1, 1), pad=(0, 0), with_relu=False,
                       name='sw_projection')
    return outputs


def rnn_northwest(local_features, height, width, num_hidden, init_state, use_gru=False):
    """
    NorthWest direction
    """
    hiddens = init_hw_matrix(height, width)
    if use_gru:
        rnn_f = gru
        param = init_gru_param('nw')
    else:
        rnn_f = rnn
        param = init_rnn_param('nw')

    for h in reversed(xrange(height)):
        for w in reversed(xrange(width)):
            loc_feat = local_features[h * width + w]
            if w == width - 1 and h == height - 1:
                prev_hidden = init_state.h
            elif w < width - 1 and h == height - 1:
                assert hiddens[h][w+1] is not None, "Hidden state is not initialized yet."
                prev_hidden = hiddens[h][w+1]
            elif w == width - 1 and h < height - 1:
                assert hiddens[h+1][w] is not None, "Hidden state is not initialized yet."
                prev_hidden = hiddens[h+1][w]
            else:
                assert hiddens[h+1][w] is not None and hiddens[h][w+1] is not None, \
                        "Hidden state is not initialized yet."
                prev_hidden = hiddens[h+1][w] + hiddens[h][w+1]

            next_state = rnn_f(num_hidden, loc_feat, prev_hidden, param, w, h, 'nw')
            hiddens[h][w] = next_state.h

    outputs = reduce(lambda x, y: x + y, hiddens) # list of symbols, [(b, c), ..]
    outputs = [mx.symbol.expand_dims(data=s, axis=2) for s in outputs]
    outputs = mx.symbol.Concat(*outputs, dim=2)
    outputs = mx.symbol.Reshape(data=outputs, shape=(0, 0, height, width))
    outputs = get_conv(data=outputs, num_filter=num_hidden, kernel=(1, 1),
                       stride=(1, 1), pad=(0, 0), with_relu=False,
                       name='nw_projection')
    return outputs


def rnn_northeast(local_features, height, width, num_hidden, init_state, use_gru=False):
    """
    NorthEast direction
    """
    hiddens = init_hw_matrix(height, width)
    if use_gru:
        rnn_f = gru
        param = init_gru_param('ne')
    else:
        rnn_f = rnn
        param = init_rnn_param('ne')

    for h in reversed(xrange(height)):
        for w in xrange(width):
            loc_feat = local_features[h * width + w]
            if w == 0 and h == height - 1:
                prev_hidden = init_state.h
            elif w > 0 and h == height - 1:
                assert hiddens[h][w-1] is not None, "Hidden state is not initialized yet."
                prev_hidden = hiddens[h][w-1]
            elif w == 0 and h < height - 1:
                assert hiddens[h+1][w] is not None, "Hidden state is not initialized yet."
                prev_hidden = hiddens[h+1][w]
            else:
                assert hiddens[h+1][w] is not None and hiddens[h][w-1] is not None, \
                        "Hidden state is not initialized yet."
                prev_hidden = hiddens[h+1][w] + hiddens[h][w-1]

            next_state = rnn_f(num_hidden, loc_feat, prev_hidden, param, w, h, 'ne')
            hiddens[h][w] = next_state.h

    outputs = reduce(lambda x, y: x + y, hiddens) # list of symbols, [(b, c), ..]
    outputs = [mx.symbol.expand_dims(data=s, axis=2) for s in outputs]
    outputs = mx.symbol.Concat(*outputs, dim=2)
    outputs = mx.symbol.Reshape(data=outputs, shape=(0, 0, height, width))
    outputs = get_conv(data=outputs, num_filter=num_hidden, kernel=(1, 1),
                       stride=(1, 1), pad=(0, 0), with_relu=False,
                       name='ne_projection')
    return outputs


def dag_rnn(conv_features, height, width, num_hidden, use_gru=False):
    """
    DAG-Recurrent Neural Networks
    Args:
        conv_features: output of conv layer.
        height: height of tensor conv_features.
        width: width of tensor conv_features.
        num_hidden: number of units used in rnn's hidden layer.
    Return:
        context_info: Global context information, which is a tensor of size (b, c, h, w)
    """
    conv_features = mx.symbol.Reshape(data=conv_features, shape=(0, 0, -1))
    conv_features = mx.symbol.SliceChannel(data=conv_features, axis=2, num_outputs=height * width,
                                           squeeze_axis=True)
    init_h = mx.symbol.Variable('init_h')
    if use_gru:
        state = GRUState(h=init_h)
    else:
        state = RNNState(h=init_h)
    # Perform RNN on each direction
    se_context_info = rnn_southeast(local_features=conv_features,
                                    height=height,
                                    width=width,
                                    num_hidden=num_hidden,
                                    init_state=state,
                                    use_gru=use_gru)
    sw_context_info = rnn_southwest(local_features=conv_features,
                                    height=height,
                                    width=width,
                                    num_hidden=num_hidden,
                                    init_state=state,
                                    use_gru=use_gru)
    ne_context_info = rnn_northeast(local_features=conv_features,
                                    height=height,
                                    width=width,
                                    num_hidden=num_hidden,
                                    init_state=state,
                                    use_gru=use_gru)
    nw_context_info = rnn_northwest(local_features=conv_features,
                                    height=height,
                                    width=width,
                                    num_hidden=num_hidden,
                                    init_state=state,
                                    use_gru=use_gru)
    # aggregate context information from all directions
    context_info = se_context_info + sw_context_info +\
                   ne_context_info + nw_context_info
    context_info = mx.symbol.LeakyReLU(name='context_prelu', act_type='prelu',
                                       data=context_info)
    return context_info


def rnn_left2right(local_features, height, width, num_hidden, init_state, use_gru=False):

    img1_feature = local_features[0]
    img2_feature = local_features[1]
    hiddens = init_hw_matrix(height, width*2)
    hiddens2 = init_hw_matrix(height, width)

    if use_gru:
        rnn_f = gru
        param_encoder = init_gru_param('left2right_encoder')
        param_decoder = init_gru_param('left2right_decoder')
    else:
        rnn_f = rnn
        param_encoder = init_rnn_param('left2right_encoder')
        param_decoder = init_rnn_param('left2right_decoder')

    for h in range(height):
        for w in range(width):
            # right image
            loc_feat = img2_feature[h * width + w]
            if w == 0 :
                prev_hidden = init_state.h
            elif w > 0 :
                assert hiddens[h][w-1] is not None, "Hidden state is not initialized yet."
                prev_hidden = hiddens[h][w-1]
            next_state = rnn_f(num_hidden, loc_feat, prev_hidden, param_encoder, w, h, 'left2right')
            hiddens[h][w] = next_state.h

        for w in range(width, 2*width):
            # left image
            loc_feat = img1_feature[h * width + w - width]
            assert hiddens[h][w - 1] is not None, "Hidden state is not initialized yet."
            prev_hidden = hiddens[h][w - 1]
            next_state = rnn_f(num_hidden, loc_feat, prev_hidden, param_decoder, w, h, 'left2right')
            hiddens[h][w] = next_state.h
            hiddens2[h][w-width] = next_state.h

    outputs = reduce(lambda x, y: x + y, hiddens2)
    outputs = [mx.symbol.expand_dims(data=s, axis=2) for s in outputs]
    outputs = mx.symbol.Concat(*outputs, dim=2)
    outputs = mx.symbol.Reshape(data=outputs, shape=(0, 0, height, width))
    outputs = get_conv(data=outputs, num_filter=num_hidden, kernel=(1, 1),
                       stride=(1, 1), pad=(0, 0), with_relu=False,
                       name='left2right_projection',bn_momentum=0.9)
    return outputs


def sequence2sequence(img1, img2, height, width, use_gru, num_hidden):

    img1_features = mx.symbol.Reshape(data=img1, shape=(0, 0, -1))
    img1_features = mx.symbol.SliceChannel(data=img1_features, axis=2, num_outputs=height * width,
                                           squeeze_axis=True)

    img2_features = mx.symbol.Reshape(data=img2, shape=(0, 0, -1))
    img2_features = mx.symbol.SliceChannel(data=img2_features, axis=2, num_outputs=height * width,
                                           squeeze_axis=True)

    init_h = mx.symbol.Variable('init_h')
    if use_gru:
        state = GRUState(h=init_h)
    else:
        state = RNNState(h=init_h)

    # Perform RNN on each direction
    out_left2right = rnn_left2right(local_features=[img1_features, img2_features],
                                    height=height,
                                    width=width,
                                    num_hidden=num_hidden,
                                    init_state=state,
                                    use_gru=use_gru)

    return out_left2right