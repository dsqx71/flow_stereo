from data import dataset, data_util
from predict.pipeline import Pipeline

output_prefix = '/home/xudong/kitti2015_test/dispnet'
piper = Pipeline('/home/xudong/model_zoo/model.config')
data = dataset.KittiDataset('stereo', '2015', is_train=False)

for index, item in enumerate(data.dirs):
    img1, img2, label, aux = data.get_data(item,'stereo')
    dis = piper.process(img1, img2)
    dis = data_util.writeKittiSubmission(dis, prefix=output_prefix, index=index, type='stereo')


