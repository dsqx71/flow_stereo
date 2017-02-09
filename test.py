
# coding: utf-8

# In[1]:

from flowstereo import pipe
import dataset
import matplotlib.pyplot as plt
from utils import util
# get_ipython().magic(u'matplotlib inline')


# In[2]:

pipeline = pipe.Pipeline('../ft_kitti/model2.config')


# In[3]:

data = dataset.FlyingChairsDataset()


# In[17]:

for item in data.dirs[10:]:
    img1,img2,label,_ = data.get_data(item,'flow')
    ret = pipeline.process(img1,img2)
    util.flow2color(ret)
    plt.waitforbuttonpress()
    util.plot_velocity_vector(ret)
    plt.waitforbuttonpress()
    
    util.flow2color(label)
    plt.waitforbuttonpress()
    util.plot_velocity_vector(label)
    plt.waitforbuttonpress()
    break


# In[23]:

# get_ipython().magic(u'pinfo2 pipeline')


# In[ ]:



