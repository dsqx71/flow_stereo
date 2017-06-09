import cv2
import skvideo.io
import time
import argparse
from .pipeline import Pipeline
from ..others import visualize

if __name__ == '__main__':

    #args
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='directory of video')
    parser.add_argument('--deploy_config', type=str, help='directory of model config')
    parser.add_argument('--fps', type=int, default=100)
    parser.add_argument('--output',type=str, help='path of output file')
    args = parser.parse_args()

    # init model
    config_path = args.deploy_config
    piper = Pipeline(config_path)

    # video setting
    video = skvideo.io.vreader(args.video_path)


    # Saving Video
    if args.output is not None:
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, 60, (768, 384), True)
    else:
        # setup cv2 windows
        cv2.namedWindow('flow', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('flow', 768, 384)

        cv2.namedWindow('vector', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('vector', 768, 384)

        cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame1', 768, 384)

    # prediction
    img1 = video.next()
    img1 = cv2.resize(img1, (768, 384))
    img1 = img1[:, :, [2, 1, 0]]
    for index, img2 in enumerate(video):
        print index
        # if index > 500:
        #     break
        # # show
        img2 = cv2.resize(img2, (768,384))
        img2 = img2[:, :, [2, 1, 0]]
        ret = piper.process(img1, img2)

        color = visualize.flow2color(ret)
        vector = visualize.flow2vector(ret, interval=10)
        color = cv2.addWeighted(img1, 0.3, color, 0.7, 0)
        vector = cv2.addWeighted(img1, 0.6, vector, 0.4, 0)

        if args.output is not None:
            out.write(vector)
        else:
            cv2.imshow('frame1', img1)
            cv2.imshow('flow', color)
            cv2.imshow('vector', vector)
            cv2.waitKey(4)
        img1 = img2

    # release
    out.release()
    cv2.destroyAllWindows()




