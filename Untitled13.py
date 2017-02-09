
# coding: utf-8

# In[1]:

import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import dataset
import cv2
from utils import util
np.set_printoptions(threshold=100,edgeitems=1000)
# get_ipython().magic(u'matplotlib inline')


# In[2]:

# img2 = mx.sym.Variable('img2')
# flow = mx.sym.Variable('flow')
# net = mx.sym.Warp(data=img2,grid=flow,only_grid=False)
# exe1 = net.simple_bind(ctx = mx.gpu(4),img2=(1,384,512,3),flow=(1,384,512,2),grad_req='write')
# exe2 = net.simple_bind(ctx = mx.cpu(0),img2=(1,384,512,3),flow=(1,384,512,2),grad_req='write')
data = dataset.FlyingChairsDataset()

# In[4]:

for item in data.dirs[7:]:
    
    img1,img2,flow,_ = data.get_data(item,'flow')
    break

img2 = mx.sym.Variable('img2')
flow = mx.sym.Variable('flow')

net = mx.sym.SpatialTransformer(data=img2,loc=flow,sampler_type='bilinear',transform_type='affine',target_shape=(384,512))


# In[6]:

exe3 = net.simple_bind(ctx = mx.gpu(7),img2=(1,3,384,512),flow=(1,6),grad_req='write')


# In[7]:

rand_num = np.random.randint(-10,10,size=(1,2))


# In[8]:

img1,img2,flow,_ = data.get_data(item,'flow')
exe3.arg_dict['img2'][:] = np.expand_dims(img2.transpose(2,0,1),0)
exe3.arg_dict['flow'][:] = np.array([[1,0,0,0,1,0]])#np.expand_dims(flow.transpose(2,0,1),0)


# In[9]:

exe3.forward()


# In[10]:

output3 = exe3.outputs[0].asnumpy()[0].transpose(1,2,0)
plt.imshow(output3)
plt.colorbar()


# In[22]:

plt.imshow(output3)
plt.colorbar()


# In[24]:

output3.min()


# In[12]:

(exe3.outputs[0].asnumpy()[0].transpose(1,2,0).astype(np.uint8) -exe3.arg_dict['img2'].asnumpy()[0].transpose(1,2,0).astype(np.uint8)).max()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



