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
    parser.add_argument('--fps', type=int, default=500)
    args = parser.parse_args()

    # init
    video = cv2.VideoCapture(args.video_dir)
    config_path = args.config_dir
    piper = Pipeline(config_path)
    flow = []

    fps = args.fps
    video.set(cv2.cv.CV_CAP_PROP_FPS,fps)
    size = (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

    cv2.namedWindow('flow',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('flow',500,800)
    cv2.namedWindow('vector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('vector', 500, 800)
    cv2.namedWindow('frame1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame1',500,800)

    data = []
    success, img1 = video.read()

    # predicting
    while success :

        begin = time.time()

        success, img1 = video.read()
        success, img2 = video.read()

        if img2 is None or img1 is None:
            break

        cv2.imshow('frame1',img1)
        ret = piper.process(img1,img2)
        flow2color(ret,is_cv2imshow=True)
        plot_velocity_vector(ret,interval=10,is_cv2imshow=True)
        cv2.waitKey(1)

        print  '{} fps'.format(1.0/(time.time() - begin))

    video.release()
    cv2.destroyAllWindows()




