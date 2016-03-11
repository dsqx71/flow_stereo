#-*- coding:utf -8
import matplotlib
matplotlib.use('Agg')
from util import get_kitty_data_dir2012,get_kitty_data_dir2015,compute_unary,\
produce_stereo_matching,outlier_sum
import pandas as pd
import mxnet  as mx
from skimage import io
import densecrf as dcrf
import numpy as np
import time
from scipy import signal
import cv2
import scipy.io as sio

if __name__ == '__main__':

	dirs = get_kitty_data_dir2012(180,194)
	dirs.extend(get_kitty_data_dir2015(180,200))
	dis_num = 227
	result = []

	for idx,img_dir in enumerate(dirs):

		begin = time.time()
		d_l,d_r,ms_l,ms_r,w1,w2 = produce_stereo_matching(img_dir,mx.gpu(3),dis_num,18)
		gt   = io.imread(img_dir[0])
		left = io.imread(img_dir[1])
		right= io.imread(img_dir[2])

		sio.savemat( '../filtercode/img_stereo/stereo_{}'.format(idx),{'ms_l':-ms_l,'ms_r':-ms_r})
		cv2.imwrite( '../filtercode/img_stereo/left_{}.ppm'.format(idx),left)
		cv2.imwrite( '../filtercode/img_stereo/right_{}.ppm'.format(idx),right)

		print img_dir[0]
		#----------------mrf-----------------------------------
		
		d = dcrf.DenseCRF2D(d_l.shape[1],d_l.shape[0],dis_num)
		u = compute_unary(ms_l,dis_num,0.7,0.0054)
		d.setUnaryEnergy(u)
		left = np.ascontiguousarray(left)
		d.addPairwiseBilateral(sxy=(15,15), srgb=(13, 13, 13), rgbim=left,
		                       compat=30,
		                       kernel=dcrf.FULL_KERNEL,
		                       normalization=dcrf.NORMALIZE_SYMMETRIC)
		Q = d.inference(4)
		dis_map = np.argmax(Q, axis=0).reshape(d_l.shape[:2])
		mrf_err,outlier = outlier_sum(dis_map,gt,3)
		print  'mrf outlier : %f' % mrf_err		
		
		#---------------median--------------------------------
		
		tmp = signal.medfilt2d(dis_map.astype(np.float32),27)
		median_err,outlier = outlier_sum(tmp,gt,3)

		print 'median filter outlier : %f' % median_err
		use_time = time.time() - begin
		
		#---------------opencv sgm----------------------------
		
		window_size = 32
		stereo = cv2.StereoSGBM(minDisparity = 0,
		        numDisparities = 128,
		        SADWindowSize= 11,
		        P1 = 5,
		        P2 = 90,
		        disp12MaxDiff = -1,
		        uniquenessRatio = 1,
		        speckleWindowSize = 100,
		        speckleRange = 128,
		        fullDP = True
		    )
		disparity = stereo.compute(left,right)
		opencv_err,outlier = outlier_sum(disparity/16.0,gt)
		print 'opencv sgbm err: %f ' % opencv_err
		result.append([img_dir[0],use_time,mrf_err,median_err,opencv_err])

result = pd.DataFrame(data=result,columns =['image_dir','time','mrf_outlier','median_filter_outlier','opencv benchmark outlier'])
result.to_csv('evaluation_result',sep='\t')

