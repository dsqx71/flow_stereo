#-*- coding:utf -8
import matplotlib
matplotlib.use('Agg')
from util import get_kitty_data_dir2012,get_kitty_data_dir2015,\
produce_stereo_matching,outlier_sum,implement_guided_filter,\
mrf_guided_filter,test_set_kitty_dir2015,mrf_dp

import matplotlib.pyplot as plt
import pandas as pd
import mxnet  as mx
from skimage import io
import numpy as np
import time
from scipy import signal
import cv2
import scipy.io as sio
import argparse


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--type',action='store',dest='type',type=int)
	cmd = parser.parse_args()
	dis_num    = 120
	pn1 = 0.2
	pn2 = 4

	if cmd.type == -1:
		#validation 
		dirs =      get_kitty_data_dir2012(180,194)
		dirs.extend(get_kitty_data_dir2015(180,200))
		result     = []
		
		for idx,img_dir in enumerate(dirs):
			
			print img_dir[0]
			begin = time.time()
			gt   = io.imread(img_dir[0])
			left = io.imread(img_dir[1])
			right= io.imread(img_dir[2])

			#compute matching score -> guided filter -> median filter
			ms_l,ms_r,w1,w2 = produce_stereo_matching(img_dir,mx.gpu(2),dis_num,18)
			dis_map  = implement_guided_filter(ms=ms_l,left=left,radius=8,epsilon= 0.7,scale=3)
			tmp = signal.medfilt2d(dis_map.astype(np.float32),9)
			dp = mrf_dp(ms_l,pn1,pn2,dis_num,left)
		
			use_time = time.time() - begin
			gf_err,outlier = outlier_sum(dis_map,gt,3)
			median_err,outlier = outlier_sum(tmp,gt,3)
			dp_err,outlier = outlier_sum(dp,gt,3)
			
			fig = plt.figure()

			tot = 5
			p1  = fig.add_subplot(tot,1,1)
			p2  = fig.add_subplot(tot,1,2)
			p3  = fig.add_subplot(tot,1,3)
			p4  = fig.add_subplot(tot,1,4)
			
			p1.imshow(left)
			p2.imshow(gt)
			p3.imshow(tmp)
			p4.imshow(dp)

			plt.savefig('./result/{}.jpg'.format(idx))
			plt.close()   

			print 'median filter outlier : %f dp outlier : %f' % (median_err,dp_err)
			result.append([img_dir[0],use_time,gf_err,median_err,dp_err])

		result = pd.DataFrame(data=result,columns =['image_dir','time','guided_filter_outlier','median_filter_outlier','dp_outlier'])
		result.to_pickle('./eval_result/evaluation_result'+ time.strftime('%Y-%m-%d-%H:%M',time.localtime(time.time())) + '.pickle' )
		result.to_csv('./eval_result/evaluation_result'+ time.strftime('%Y-%m-%d-%H:%M',time.localtime(time.time())) ,sep='\t')
		print result.describe()
	
	else:
		# test 
		dirs = test_set_kitty_dir2015(0,20)
		for idx,img_dir in enumerate(dirs):
			
			print img_dir[0]
			left = io.imread(img_dir[1])
			right= io.imread(img_dir[2])

			ms_l,ms_r,w1,w2 = produce_stereo_matching(img_dir,mx.gpu(2),dis_num,18)
			dis_map  = implement_guided_filter(ms=ms_l,left=left,radius=5,epsilon= 0.001,scale=1)
			tmp = signal.medfilt2d(dis_map.astype(np.float32),7)
			dp = mrf_dp(ms_l,pn1,pn2,dis_num,left)

			fig = plt.figure()
			p1 = fig.add_subplot(3,1,1)
			p2 = fig.add_subplot(3,1,2)
			p3 = fig.add_subplot(3,1,3)

			
			p1.imshow(left)
			p2.imshow(tmp)
			p3.imshow(dp)

			plt.savefig('./test_result/{}.jpg'.format(idx))
			plt.close()   

