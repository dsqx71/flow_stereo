import stereovision.calibration as calibration
import cv2

# load camera matrix
calib = calibration.StereoCalibration(input_folder='/rawdata/stereo/calibration/')

#load original image
img1 = cv2.imread('/rawdata/stereo/test6/i13370_cam2_1432991539.103002.jpg')
img2 = cv2.imread('/rawdata/stereo/test6/i13370_cam1_1432991539.098778.jpg')

#size of image we calibrate is (768,1024)
img1 = cv2.resize(img1,(1024,768))
img2 = cv2.resize(img2,(1024,768))
# tmp1,tmp2 are the result
tmp1, tmp2 = calib.rectify((img1,img2))