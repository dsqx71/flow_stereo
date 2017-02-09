import dataiter
import dataset
import matplotlib.pyplot as plt

data_set = dataset.SythesisData('stereo', ['flyingthing3d'])

for item in data_set.dirs[1000:]:
    img1,img2,dis,_ = data_set.get_data(item,'stereo')
    plt.figure()
    plt.imshow(img1)
    plt.waitforbuttonpress(1)
    
    plt.figure()
    plt.imshow(img2)
    plt.waitforbuttonpress(1)
    
    plt.figure()
    plt.imshow(dis)
    plt.waitforbuttonpress(1)
    