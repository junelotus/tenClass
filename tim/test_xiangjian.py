import numpy as np
import cv2
import os
import glob
import os.path
from pathlib import Path
import shutil
import datetime
import testbk
from skimage.measure import  compare_ssim
import time 
def showImg(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, img)
    cv2.waitKey(1000)

def thresholdImage(img_to_correct):
    img_result_1 = cv2.cvtColor(img_to_correct,cv2.COLOR_RGB2GRAY)
    rct, img_result_1 = cv2.threshold(img_result_1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img_result_1
path = glob.glob('result/*jpg')
def to_full(img1):
	shape = img1.shape
	begin_x = 0
	end_x = shape[0]
	begin_y = 0
	end_y = shape[1]
	shape = img1.shape
	for i in range(img1.shape[0]):
			if np.bincount(np.array(img1[i]))[-1] == img1.shape[1]:
				begin_x = i
				#print('begin_x',begin_x)
			else:
				 break
	for i in range(img1.shape[0]):
			if np.bincount(np.array(img1[img1.shape[0]-i-1]))[-1] == img1.shape[1]:
				end_x = shape[0]-i
				#print('end_x',end_x)
			else:
				 break

	for i in range(img1.shape[1]):
			if np.bincount(np.array(img1[:,i]))[-1] == img1.shape[0]:
				begin_y = i
				#print('begin_x',begin_x)
			else:
				 break
	for i in range(img1.shape[1]):
			if np.bincount(np.array(img1[:,img1.shape[1]-i-1]))[-1] == img1.shape[0]:
				end_y = shape[1]-i
				#print('end_x',end_x)
			else:
				 break

	img1 = img1[begin_x:end_x,begin_y:end_y]
	size = (shape[1], shape[0])
	img1 = cv2.resize(img1, size, interpolation=cv2.INTER_LINEAR)
	return img1

for ii in range(int(len(path)/2)):
	if ii == 0:
		continue
	img1 = cv2.imread('result/img_'+str(ii)+'.jpg')
	img2 = cv2.imread('result/model_'+str(ii)+'.jpg')
	if img1.ndim==3:
		img1 = thresholdImage(img1)
		begin_x = 0
		end_x  = 0
		begin_y = 0
		end_y  = 0
		# img1 = to_full(img1)
	if img2.ndim==3:
		img2 = thresholdImage(img2)
		# img2 = to_full(img2)
	# #print(img1)
	shape  = img1.shape
	counter = 0
	first = 0
	second = 0
	for i  in range(shape[0]):
		for j in range(shape[1]):
			if img1[i][j] == img2[i][j]:
				img1[i][j]=255#[255,255,255]

			else :
				if img1[i][j] == 255:
					first = first+1
				else :
					second = second+1
				counter = counter+1
				img1[i][j]=0#[0,0,0]
	contours, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	# for k in range(len(contours)):
	# 	cv2.drawContours(img1, contours, k, 0, cv2.FILLED)
	# showImg('mask',img1)
	pre_img = img1
	# if  counter/(shape[0]*shape[1]) > 0.05 and first/(shape[0]*shape[1])>=0.05 :#and min(first,second)*2<max(first,second):
	# 	if ii <=1:
	# 		for i in range(img1.shape[0]):
	# 			for j in range(img1.shape[1]):
	# 				if img1[i][j] == 0 and img1[i][j] == pre_img[i][j]:
	# 					img1[i][j] = 255
	# 		pre_img = img1
	# 	else:
	# 		pre_img = img1
		
	cv2.imwrite("result2/result_"+str(ii)+'_'+str(counter)+'_'+str(first)+'_'+str(second)+".jpg",img1)
	#print(counter)
