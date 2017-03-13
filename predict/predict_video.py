import cv2
import time
import argparse
from .pipeline import Pipeline
from others import visualize

if __name__ == '__main__':

    #args
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, help='directory of video')
    parser.add_argument('--config_dir', type=str, help='directory of model config')
    parser.add_argument('--fps', type=int, default=100)
    args = parser.parse_args()

    # init model
    config_path = args.config_dir
    piper = Pipeline(config_path)

    # video setting
    video = cv2.VideoCapture(args.video_dir)
    video.set(cv2.cv.CV_CAP_PROP_FPS, args.fps)

    # setup cv2 windows
    cv2.namedWindow('flow', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('vector', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('frame1', cv2.WINDOW_AUTOSIZE)

    # prediction
    success, img1 = video.read()
    while success:
        success, img2 = video.read()
        if ~success:
            break
        # show
        cv2.imshow('frame1', img1)
        ret = piper.process(img1, img2)
        color = visualize.flow2color(ret)
        cv2.imshow('flow', color)
        vector = visualize.flow2vector(ret, interval=20)
        cv2.imshow('vector', vector)
        cv2.waitKey(1)
        img1 = img2

    # release
    video.release()
    cv2.destroyAllWindows()




