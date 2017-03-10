import argparse
from ..data import dataset, data_util
from .pipeline import Pipeline
from ..others import visualize

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, help='directory of model config')
    parser.add_argument('--output_prefix', type=str, help='prefix of output directory')
    parser.add_argument('--data_type', type=str, help='data type', choices=['stereo','flow'])
    parser.add_argument('--is_show', type=bool, help='whether show results', default=False)
    args = parser.parse_args()

    # init pipeline
    piper = Pipeline(args.model_config)
    data = dataset.KittiDataset(args.data_type, '2015', is_train=False)

    # prediction
    for index, item in enumerate(data.dirs):
        img1, img2, label, aux = data.get_data(item)
        dis = piper.process(img1, img2)
        if args.is_show:
            visualize.plot_pairs(img1, img2, dis, 'stereo', plot_patch=False)
        data_util.writeKittiSubmission(dis, prefix=args.output_prefix, index=index, type=args.data_type)


