import mxnet as mx
import rnn_unit
from symbol.dispnet_symbol import get_loss

def stereo_rnn(height, width, use_gru, num_hidden):

    img1 = mx.sym.Variable('img0_feature_output')
    img2 = mx.sym.Variable('img1_feature_output')
    label  = mx.sym.Variable('stereo_downsample1')
    rnn_output = rnn_unit.sequence2sequence(img1, img2, height, width, use_gru, num_hidden)

    pr = mx.sym.Convolution(rnn_output, pad = (2,2),kernel=(5,5),stride=(1,1),num_filter = 1,name='predict')
    loss = get_loss(data=pr,label=label,grad_scale=1,name='loss')
    return loss