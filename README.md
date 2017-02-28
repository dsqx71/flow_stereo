#FlowNet & DispNet
The architectures of flownet and dispnet are the same ,except for output ndim.
Rotation will destroy epipolar line,so users are prohibited to rotate stereo data

---
### Example

``
	python train.py --type stereo --continue  which_checkpoint  --lr learning_rate
``

### Option:

* Most options defined in config.py

### Prediction：

* please refer to prediction example.ipynb


###Data organization
 - Class **Dataset** : if you want to add new dataset,please refer to **Dataset** and provide the following functions
 	- *init* : provide the directory of the dataset.
 	- *shapes* : the output of Dispnet is sensitive to the original shapes,so you need to  provide them.
 	- *get_data* : how to get the data

 - **Iterators**: We provide two kinds of data iterator,they are independent of specific dataset
 	- Class **Dataiter** :self-defined iterator based on python
 	- Class **multi_imageRecord** : a wrapper of multi image records,If you want to tranfer stereo data to .rec file,a tool in util.py will help you.
 	- Iterators provide augmentation and some preprocessing like padding.
 	- If is_train is false,it will not augment data ,and if label exists it will provide original labels





## Getting Started

- Build extension: `make all`
- Get symbol link: `ln -s kitti_dir ./data/kitti`

## Dataset decription

#### KITTI
- [The dataset description](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
- KITTI stereo/flow contains two part: KITTI2015 and KITTI2012
- If you want to know details of the dataset organization, please refer to the following:
 - data/KITTI2015.txt
 - data/KITTI2012.txt
- Assume you place two parts in the same folder: /rawdata/stereo/kitti/

#### Scene Flow Datasets(Synthesis dataset)
- [The dataset description](http://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- There is no official description of dataset organization. But you can refer to dataset.py or [the caffe code](http://lmb.informatik.uni-freiburg.de/resources/binaries/dispflownet/dispflownet-release-1.2.tar.gz).
- Scene Flow Datasets contains three different scenes:
	- Driving
	- Flyingthing
	- Monkaa
- Assume you place three parts in the same folder: /rawdata/stereo/

####The "Flying Chairs" dataset
- [The dataset description](http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)
- The dataset organization is very simple:
	- first frame :  xxxxx_img1.ppm
	- second frame : xxxxx_img2.ppm
	- optical flow : xxxxx_flow.flo


## tools
we provide the following functions in utils：

 - weighted median filter
 - flow2color : plot optical flow
 - readPFM : how to read .pfm file
 - plot_velocity_vector : another way to plot optical flow，more intuitive.
 - get_imageRecord: transfer the dataset to .rec file








    




