## Stereo and Optical flow


#### Installation Guide
- Install dependencies: numpy, opencv2, MXNet. 
- If the MXNet version which you used doesn't support correlation1D operator, you should install MXNet with additional operatior -- edit ```EXTRA_OPERATORS``` in ```config.mk``` to include the ```flow_stereo/operator``` folder.
- Run ```make all``` to build extension
- Edit all fields in data/config.py, otherwise it will not work.

#### Project Organization
- ```data.dataloader```: Data Loader
- ```data.dataset```: Dataset class
- ```data.augmentation```: Pipeline for augmentation
- ```predict.pipeline```: Pipeline for prediction
- ```docs```: Dataset documents
- ```symbol```: Symbols and customed operators
- ```others```: visualization, metric, and other utilities

#### Usage
- For training, run ```python -m flow_stereo.train exp_name EXPERIMENT_NAME --gpus GPU_INDEX --epoch RESUMING_EPOCH```