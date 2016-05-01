#FlowNet & DispNet

The architectures of flownet and dispnet are the same ,except for output ndim.

Rotation will destroy epipolar line,so users are prohibited to rotate stereo data

---
### Example

``
python train.py --type stereo --continue  which_checkpoint  --lr learning_rate  --ctx contex
``

### Option:

* Most options defined in config.py
* If you want to add new dataset or change setting , please refer to dataset.py


### Prediction：

* please refer to prediction example.ipynb.
* multiprocessing will lead ipython notebook kernel crash,if you want to use multi-process,please convert the example to .py file
