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


###Data  organization
 - Class **Dataset** : if you want to add new dataset,please refer to **Dataset** and provide the following functions
 	- *init* : provide the directory of the dataset.
 	- *shapes* : the output of Dispnet is sensitive to the original shapes,so you need to  provide them.
 	- *get_data* : how to get the data
 
 - **Iterators**: We provide two kinds of data iterator,they are independent of specific dataset
 	- Class **Dataiter** :self-defined iterator based on python 
 	- Class **multi_imageRecord** : a wrapper of multi image records,If you want to tranfer stereo data to .rec file,a tool in util.py will help you.
 	- Iterators provide augmentation and some preprocessing like padding.
 	- If is_train is false,it will not augment data ,and if label exists it will provide original labels
 
### tools
we provide the following functions in util.py：

 - weighted median filter 
 - flow2color : plot optical flow
 - readPFM : how to read .pfm file
 - plot_velocity_vector : another way to plot optical flow，more intuitive.
 - get_imageRecord: transfer the dataset to .rec file





























    




