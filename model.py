import mxnet as mx


def flow_and_stereo_net(net_type='flow', loss1_scale = 0.003, loss2_scale = 0.005, loss3_scale = 0.01,loss4_scale= 0.02,loss5_scale = 0.08,loss6_scale = 0.32):
    """
        Dispnet: A large Dataset to train Convolutional networks for disparity, optical flow,and scene flow estimation

        The  architectures of dispnet and flownet are the same

        loss_scale : the weight of loss layer
    """
    if net_type == 'flow':
        output_dim = 2
    elif net_type == 'stereo' :
        output_dim = 1

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')

    # labels with different shape

    downsample1 = mx.sym.Variable(net_type + '_downsample1')
    downsample2 = mx.sym.Variable(net_type + '_downsample2')
    downsample3 = mx.sym.Variable(net_type + '_downsample3')
    downsample4 = mx.sym.Variable(net_type + '_downsample4')
    downsample5 = mx.sym.Variable(net_type + '_downsample5')
    downsample6 = mx.sym.Variable(net_type + '_downsample6')

    concat1 = mx.sym.Concat(img1,img2,dim=1,name='concat_image')

    conv1   = mx.sym.Convolution(concat1, pad = (3,3), kernel=(7,7),stride=(2,2),num_filter=64,name='conv1')
    conv1   = mx.sym.LeakyReLU(data = conv1,  act_type = 'leaky', slope  = 0.1 )

    conv2   = mx.sym.Convolution(conv1,  pad  = (2,2),  kernel=(5,5),stride=(2,2),num_filter=128,name='conv2')
    conv2   = mx.sym.LeakyReLU(data = conv2, act_type =  'leaky', slope  = 0.1 )

    conv3a   = mx.sym.Convolution(conv2, pad  = (2,2), kernel=(5,5),stride=(2,2),num_filter=256,name='conv3a')
    conv3a   = mx.sym.LeakyReLU(data = conv3a,  act_type = 'leaky',slope  = 0.1 )

    conv3b   = mx.sym.Convolution(conv3a,pad = (1,1) , kernel=(3,3),stride=(1,1),num_filter=256,name='conv3b')
    conv3b   = mx.sym.LeakyReLU(data = conv3b ,act_type = 'leaky',slope = 0.1 )

    conv4a   = mx.sym.Convolution(conv3b,pad=(1,1),kernel=(3,3),stride=(2,2),num_filter=512,name='conv4a')
    conv4a  = mx.sym.LeakyReLU(data = conv4a,act_type = 'leaky',slope  = 0.1 )

    conv4b = mx.sym.Convolution(conv4a,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter=512,name='conv4b')
    conv4b   = mx.sym.LeakyReLU(data = conv4b,act_type = 'leaky',slope  = 0.1 )

    conv5a  = mx.sym.Convolution(conv4b,pad=(1,1),kernel=(3,3),stride=(2,2),num_filter=512,name='conv5a')
    conv5a  = mx.sym.LeakyReLU(data = conv5a,act_type = 'leaky',slope  = 0.1 )

    conv5b = mx.sym.Convolution(conv5a,pad= (1,1),kernel=(3,3),stride=(1,1),num_filter=512,name='conv5b')
    conv5b  = mx.sym.LeakyReLU(data = conv5b,act_type = 'leaky',slope  = 0.1 )

    conv6a = mx.sym.Convolution(conv5b,pad= (1,1),kernel=(3,3),stride=(2,2),num_filter=1024,name='conv6a')
    conv6a  = mx.sym.LeakyReLU(data = conv6a,act_type = 'leaky',slope  = 0.1 )

    conv6b = mx.sym.Convolution(conv6a,pad= (1,1),kernel=(3,3),stride=(1,1),num_filter=1024,name='conv6b')
    conv6b  = mx.sym.LeakyReLU(data = conv6b,act_type = 'leaky',slope  = 0.1, )

    pr6 = mx.sym.Convolution(conv6b,pad= (1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name='pr6')

    loss6 = mx.sym.MAERegressionOutput(data = pr6,label = downsample6,grad_scale=loss6_scale,name='loss6')

    upsample_pr6to5 = mx.sym.Deconvolution(pr6,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=2,name='upsample_pr6to5')
    upsample_pr6to5 = mx.sym.LeakyReLU(data = upsample_pr6to5,act_type = 'leaky',slope  = 0.1 )

    upconv5 = mx.sym.Deconvolution(conv6b,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=512,name='upconv5')
    upconv5 = mx.sym.LeakyReLU(data = upconv5,act_type = 'leaky',slope  = 0.1  )
    concat_tmp = mx.sym.Concat(upconv5,upsample_pr6to5,conv5b,dim=1)

    iconv5 = mx.sym.Convolution(concat_tmp,pad = (1,1),kernel=(3,3),stride=(1,1),num_filter = 512,name='iconv5')
    iconv5 = mx.sym.LeakyReLU(data = iconv5,act_type = 'leaky',slope  = 0.1 )

    pr5    = mx.sym.Convolution(iconv5, pad = (1,1),kernel=(3,3),stride=(1,1),num_filter = output_dim,name='pr5')
    loss5 = mx.sym.MAERegressionOutput(data = pr5,label = downsample5,grad_scale=loss5_scale,name='loss5')

    upconv4 = mx.sym.Deconvolution(iconv5,pad = (1,1),kernel= (4,4),stride = (2,2),num_filter=256,name='upconv4')
    upconv4 = mx.sym.LeakyReLU(data = upconv4,act_type = 'leaky',slope  = 0.1 )

    upsample_pr5to4 = mx.sym.Deconvolution(pr5,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=2,name='upsample_pr5to4')
    upsample_pr5to4 = mx.sym.LeakyReLU(data = upsample_pr5to4,act_type = 'leaky',slope  = 0.1 )

    concat_tmp2 = mx.sym.Concat(upsample_pr5to4,upconv4,conv4b)
    iconv4  = mx.sym.Convolution(concat_tmp2,pad = (1,1),kernel = (3,3),stride=(1,1),num_filter=256,name='iconv4')
    iconv4 =  mx.sym.LeakyReLU(data = iconv4,act_type = 'leaky',slope  = 0.1  )

    pr4 = mx.sym.Convolution(iconv4,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name='pr4')
    loss4 = mx.sym.MAERegressionOutput(data = pr4,label = downsample4,grad_scale=loss4_scale,name='loss4')

    upconv3 = mx.sym.Deconvolution(iconv4,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=128,name='upconv3')
    upconv3 = mx.sym.LeakyReLU(data = upconv3,act_type = 'leaky',slope  = 0.1 )

    upsample_pr4to3 = mx.sym.Deconvolution(pr4,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=2,name='upsample_pr4to3')
    upsample_pr4to3 = mx.sym.LeakyReLU(data = upsample_pr4to3,act_type = 'leaky',slope  = 0.1 )

    concat_tmp3 = mx.sym.Concat(upsample_pr4to3,upconv3,conv3b)
    iconv3 = mx.sym.Convolution(concat_tmp3,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter = 128,name='iconv3')
    iconv3 = mx.sym.LeakyReLU(data = iconv3,act_type = 'leaky',slope  = 0.1 )

    pr3 = mx.sym.Convolution(iconv3,pad = (1,1), kernel = (3,3), stride = (1,1),num_filter = output_dim,name='pr3')
    loss3 = mx.sym.MAERegressionOutput(data = pr3,label = downsample3,grad_scale=loss3_scale,name='loss3')

    upconv2 = mx.sym.Deconvolution(iconv3,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter=64,name='upconv2')
    upconv2 = mx.sym.LeakyReLU(data = upconv2,act_type = 'leaky',slope  = 0.1  )

    upsample_pr3to2 = mx.sym.Deconvolution(pr3,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=2,name='upsample_pr3to2')
    upsample_pr3to2 = mx.sym.LeakyReLU(data = upsample_pr3to2,act_type = 'leaky',slope  = 0.1 )
    concat_tmp4 = mx.sym.Concat(upsample_pr3to2,upconv2,conv2)

    iconv2 = mx.sym.Convolution(concat_tmp4,pad = (1,1),kernel = (3,3),stride= (1,1),num_filter = 64,name='iconv2')
    iconv2 = mx.sym.LeakyReLU(data = iconv2,act_type = 'leaky',slope  = 0.1 )
    pr2 = mx.sym.Convolution(iconv2,pad = (1,1),kernel=(3,3),stride = (1,1),num_filter = output_dim,name='pr2')
    loss2 = mx.sym.MAERegressionOutput(data = pr2,label = downsample2,grad_scale=loss2_scale,name='loss2')

    upconv1 = mx.sym.Deconvolution(iconv2,pad=(1,1),kernel=(4,4),stride=(2,2),num_filter = 32,name='upconv1')
    upconv1 = mx.sym.LeakyReLU(data = upconv1,act_type = 'leaky',slope  = 0.1 )

    upsample_pr2to1 = mx.sym.Deconvolution(pr2,pad = (1,1),kernel= (4,4),stride=(2,2),num_filter=2,name='upsample_pr2to1')
    upsample_pr2to1 = mx.sym.LeakyReLU(data = upsample_pr2to1,act_type = 'leaky',slope  = 0.1 )

    concat_tmp5 = mx.sym.Concat(upsample_pr2to1,upconv1,conv1)
    iconv1 = mx.sym.Convolution(concat_tmp5,pad=(1,1),kernel = (3,3),stride=(1,1),num_filter=32,name='iconv1')
    iconv1 = mx.sym.LeakyReLU(data = iconv1,act_type = 'leaky',slope  = 0.1  )

    pr1 = mx.sym.Convolution(iconv1,pad=(1,1),kernel=(3,3),stride=(1,1),num_filter=output_dim,name='pr1')
    loss1 = mx.sym.MAERegressionOutput(data = pr1,label = downsample1,grad_scale=loss1_scale,name='loss1')

    # dispnet and flownet have 6 L1 loss layers
    net = mx.sym.Group([loss1,loss2,loss3,loss4,loss5,loss6])

    return net