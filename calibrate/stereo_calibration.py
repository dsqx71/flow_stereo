import stereovision.calibration as calibration
import glob
import cv2
import matplotlib.pyplot as plt


dirs = glob.glob('/rawdata/stereo/test7/calibrate_img/*.jpg')
dirs.sort()
count = 0

# parameters： chessboard size ,chessboard squard size(not important),image_size
calibrator = calibration.StereoCalibrator(6, 5, 2,(1024,768))
    
for i in range(len(dirs)/2):

    right = dirs.pop(0)
    left  = dirs.pop(0)
    
    img1 = cv2.imread(left)
    img2 = cv2.imread(right)
    
    calibrator.add_corners((img1, img2))

calibration = calibrator.calibrate_cameras()
avg_error = calibrator.check_calibration(calibration)

#save camera matrix
calibration.export('/rawdata/stereo/calibration')
print ("calibration err %f" % avg_error)


