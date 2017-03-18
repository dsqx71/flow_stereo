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
- ```symbol```: Network symbols and customed operators
- ```others```: visualization, metric, and other utilities

####Training
Run ```python -m flow_stereo.train exp_name EXPERIMENT_NAME --gpus GPU_INDEX --epoch RESUMING_EPOCH```


####Testing Example
Please refer to ```predict/predict_kitti.py``` and ```predict/predict_video.py```.
Those examples provides a step-by-step walkthrough to help you learn the usage of prediction pipeline
