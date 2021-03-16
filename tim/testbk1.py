import tensorflow as tf
import glob # 获取路径
import tensorflow.keras as keras
from matplotlib import pyplot as plt # 绘图
AUTOTUNE = tf.data.experimental.AUTOTUNE # prefetch 根据CPU个数创建读取线程
import os
import math
import numpy as np
import cv2
class DetectROI:
	img = ''
	def __init__(self,net_path=None):
		self.net = tf.keras.models.load_model(net_path)

	def set_path_or_image(self,img=None,model=None):
		DetectROI.img = img
		DetectROI.model = model
	'''
	def findTwo(self,first_rec,img):
		first_rec.sort(key=self.takeX)
		# print(first_rec)
		firxt_x_min = first_rec[0][0]
		first_rec.sort(key=self.takeY)
		# print(first_rec)
		firxt_y_min = first_rec[0][1]
		first_rec.sort(key=self.takeXW)
		# print(first_rec)
		# print(first_rec)
		firxt_x_max = first_rec[-1][0]+first_rec[-1][2]
		first_rec.sort(key=self.takeYH)
		# print(first_rec)
		firxt_y_max = first_rec[-1][1]+first_rec[-1][3]
		w_now = firxt_x_max - firxt_x_min
		y_now = firxt_y_max - firxt_y_min
		shape = img.shape
		pt1 = [max(0,int(firxt_x_min-(w_now/2.5))),max(0,int(firxt_y_min-(y_now/2)))]
		pt2 = [min(firxt_x_max+int(w_now/2.5),shape[1]),min(shape[0],firxt_y_max+int(y_now/2))]
		cv2.rectangle(img, pt1=(max(0,int(firxt_x_min-(w_now/2.5))),max(0,int(firxt_y_min-(y_now/2)))), pt2=(min(firxt_x_max+int(w_now/2.5),shape[1]),min(firxt_y_max+int(y_now/2),shape[0])),color=(0, 0, 255), thickness=2)
		# return_imgs[0][0] = img[firxt_x_min:firxt_x_min]
		return pt1,pt2,img[pt1[1]:pt2[1],pt1[0]:pt2[0]]
	'''

	def findTwo(self,first_rec,img):
		first_rec.sort(key=self.takeX)
		# print(first_rec)
		firxt_x_min = first_rec[0][0]
		first_rec.sort(key=self.takeY)
		# print(first_rec)
		firxt_y_min = first_rec[0][1]
		first_rec.sort(key=self.takeXW)
		# print(first_rec)
		# print(first_rec)
		firxt_x_max = first_rec[-1][0]+first_rec[-1][2]
		first_rec.sort(key=self.takeYH)
		# print(first_rec)
		firxt_y_max = first_rec[-1][1]+first_rec[-1][3]

		w_now = int(firxt_x_max - firxt_x_min)
		w_now_bk = w_now
		y_now = firxt_y_max - firxt_y_min# w_now*2#

		if len(first_rec)==1:
			print('11111')
			w_now = int(w_now*4.5)
			y_now = w_now*1.5#y_now*10
		elif len(first_rec)==2:
			print('22222')
			w_now = w_now*3
			y_now = w_now*1.5
		elif y_now/w_now < 1.5:
			print('33333')
			w_now =int(w_now*1.5)
			y_now = y_now*1.5

		if w_now < 50:
			w_now = w_now*10
			if y_now/w_now < 1.5:
				y_now = w_now*1.5
		elif w_now < 100:
			w_now = w_now*5
			if y_now/w_now < 1.5:
				y_now = w_now*1.5
		elif w_now < 150:
			w_now = int(w_now*4)
			if y_now/w_now < 1.5:
				y_now = w_now*1.5		
		elif w_now < 200:
			w_now = int(w_now*2.5)
			if y_now/w_now < 1.5:
				y_now = w_now*1.5

		print(y_now/w_now)

		# # y_now = y_now*(w_now/)
		# if y_now/w_now_bk < 0.5:
		# 	print('55555')
		# 	y_now = w_now_bk*5
		# elif y_now/w_now_bk < 0.7:
		# 	print('7777')
		# 	y_now = w_now_bk*2.5
		# elif y_now/w_now_bk < 0.8:
		# 	print('88888')
		# 	y_now = w_now_bk*1.5

		
		
		shape = img.shape
		pt1 = [max(0,int(firxt_x_min-(w_now/3))),max(0,int(firxt_y_min-(y_now/2)))]
		pt2 = [min(firxt_x_max+int(w_now/3),shape[1]),min(shape[0],firxt_y_max+int(y_now/2))]
		# cv2.rectangle(img, pt1=(max(0,int(firxt_x_min-(w_now/3))),max(0,int(firxt_y_min-(y_now/2)))), pt2=(min(firxt_x_max+int(w_now/3),shape[1]),min(firxt_y_max+int(y_now/2),shape[0])),color=(0, 0, 255), thickness=2)
		return pt1,pt2,img[pt1[1]:pt2[1],pt1[0]:pt2[0]]

	def takeX(self,cnt):
			return cnt[0]
	def takeY(self,cnt):
			return cnt[1]
	def takeXW(self,cnt):
			return cnt[0]+cnt[2]
	def takeYH(self,cnt):
			return cnt[1]+cnt[3]
	# def jifentu(img):
	# 	shape = img.shape
	# 	for i in range 
	# 	for i in range(shape[0]):
	# 		for j in range(shape[1]):
	# 			img[i][j] = img[i][j]+img[i-1][j-1]

	def read_jpg_gt(self,path):
		img = tf.io.read_file(path)# 2.0里面的读取都在io里面
		img = tf.image.decode_jpeg(img,channels = 1)
		return img
	def load_img_label(self,path):
		##print(path)
		img = self.read_jpg_gt(path)
		img = tf.image.resize(img,(64,64))
		img = tf.cast(img, tf.float32)/127.5 - 1#img/127.5-1
		return img


	def showImg(self,name,img):
		cv2.namedWindow(name,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
		cv2.imshow(name,img)
		cv2.waitKey(20000)

	#showImg('name',cv2.imread('./score_map/1.jpg'))
	# score_label = glob.glob('./score_map/*jpg')
	# label_score_img = tf.data.Dataset.from_tensor_slices(score_label)
	# label_score_data = label_score_img.map(load_img_label,num_parallel_calls = AUTOTUNE).cache().batch(batch_size)
	# for score in label_score_data:
	# 	###print(score)
	# 	break
	def read_jpg(self,path):
		img = tf.io.read_file(path)# 2.0里面的读取都在io里面
		img = tf.image.decode_jpeg(img,channels = 3)
		return img

	def load_img_train(self):
		# img = read_jpg(path)

		img = tf.image.resize(DetectROI.img,(512,512)) # resize 会使得在后续显示的时候不是none none
		img = tf.cast(img, tf.float32)/127.5 - 1#img/127.5-1
		return img

	def tempCompare(self,target,tpl_color):
		# tpl_color = cv2.imread('model_8.jpg',cv2.IMREAD_COLOR)# np.zeros((tpl.shape[0], tpl.shape[1]),dtype=np.int8)
		# target =[img_color.copy()[0:img_color.shape[0],0:int(img_color.shape[1]/2)],img_color.copy()[0:img_color.shape[0],int(img_color.shape[1]/3):img_color.shape[1]]]

		methods = [cv2.TM_SQDIFF_NORMED]#, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]
		th, tw = tpl_color.shape[:2]
		md = cv2.TM_CCORR_NORMED
		img_result = []
		threshold = 300
		# for target_sub in target:
		result = cv2.matchTemplate(self.model, tpl_color, md)
			
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
		return max_val
			# # print('result',min_val, max_val, min_loc, max_loc )
			# if md == cv2.TM_SQDIFF_NORMED:
			# 	tl = min_loc
			# else:
			# 	tl = max_loc
			
			# py = max(0,tl[0]-threshold)
			# px = max(0,tl[1]-threshold)
			# br = (min(tl[0] + tw+threshold,target_sub.shape[1]), min(tl[1] + th+threshold,target_sub.shape[0]))
			# # cv2.rectangle(target_sub, (py,px), br, [0, 0, 255])
			# img_result.append(target_sub[ px:br[1],py:br[0]])
			

	def filterRecList(self,first_rec,xory = ['x','y']):
		# target =[img_color.copy()[0:img_color.shape[0],0:int(img_color.shape[1]/2)],img_color.copy()[0:img_color.shape[0],int(img_color.shape[1]/3):img_color.shape[1]]]
		for t in xory:
			if t == 'x':
				flag = 0
				first_rec.sort(key=self.takeX)
				threshold = 300
			else :
				flag = 1
				first_rec.sort(key=self.takeY)
				threshold = 40
			print('before first_rec',first_rec)
			max_x = -1
			index = -1
			for i in range(len(first_rec)):
				if i == 0:
					continue
				else :
					if max_x < first_rec[i][flag]-first_rec[i-1][flag]:
						max_x = first_rec[i][flag]-first_rec[i-1][flag]
						index = i
					if max_x > 200:
						# max_i = self.tempCompare(self.model,self.img[first_rec[i][1]:first_rec[i][1]+first_rec[i][3],first_rec[i][0]:first_rec[i][0]+first_rec[i][2]])
						# max_i_1 = self.tempCompare(self.model,self.img[first_rec[i-1][1]:first_rec[i-1][1]+first_rec[i-1][3],first_rec[i-1][0]:first_rec[i-1][0]+first_rec[i-1][2]])
						# 比较和模版的匹配度

						# if max_i-1 > max_i_1:
							first_rec = first_rec[:i]
						# else :
						# 	first_rec = first_rec[i:]
							break
			print('before first_rec',first_rec)
		return first_rec

	def getTwoOrOne(self,img_ori,get_one=False):
		# path  = './jpg17/IMG_20200709_110032.jpg'#'/Users/xhj/Downloads/notclear/001.jpg'#'./jpg15/danmu2.jpg'# 'danmu4.jpg'#'test_ocr.jpg'#'./jpg17/IMG_20200709_110101.jpg'#'./jpg15/danmu2.jpg'# 'join.zego.jpg'#
		img_ori_to_get_shape = img_ori.shape
		# img_ori = glob.glob(path)#('./1_ch4_training_images/img_1.jpg')#
		batch_size  = 1
		train_img = tf.data.Dataset.from_tensor_slices(img_ori)
		batch_size = 1
		train_data = train_img.map(DetectROI.load_img_train,num_parallel_calls = AUTOTUNE).cache().batch(batch_size)#.shuffle(buffer_size)
		net = tf.keras.models.load_model('net.h5')
		for t in train_data:
			pred_score,angle_map,geo_map = net(t,training=False)
			##print(geo_map.shape[0])
			size_t =64
			pred_img = np.array(tf.reshape(pred_score,(64,64)))
			shape = geo_map.shape
			t = np.array(tf.reshape(geo_map,shape))
			t = tf.squeeze(t,axis=0)
			t = tf.squeeze(t,axis=-1)
			shape = t.shape[1]
			##print('t.shape',t[1,1,:].shape)
			##print('shape',shape)
			list_cor = []

			for i in  range(shape):
				for j in range(shape):
					if  np.sum(np.array(t[i,j,:])) > 0.0:
						##print('i,j',np.sum(np.array(t[i,j,:])))#np.array(t[i,j,:]))
						position = np.array(t[i,j,:])
						# if len(list_cor) == 0 or (list_cor[-1].all()!=position.all()):
						list_cor.append(position)

			img = np.array(tf.reshape(t[:,:,0],(size_t,size_t)))

			# list_cor = list_cor

			img = img_ori#cv2.imread(path)
			img = cv2.resize(img, (img_ori_to_get_shape[1], img_ori_to_get_shape[0]))
			
			pred_img = cv2.resize(pred_img, (img_ori_to_get_shape[1], img_ori_to_get_shape[0]))
			_, pred_img = cv2.threshold(pred_img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			

			contours, _ = cv2.findContours(pred_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
			i = 0
			all_rec = []
			# print('contours',contours)
			for cnt in contours:
				x, y, w, h =  cv2.boundingRect(cnt)
				imgsub = img[y:y+h,x:x+w]

				# imgsub = cv2.bitwise_not(imgsub)
				cv2.imencode('.jpg', imgsub)[1].tofile('danmu/'+str(i)+'.jpg')
				if w < 30 or h < 30:
					continue
				all_rec.append((x,y,w,h))
				i = i+1
			all_rec.sort(key=self.takeX)
			img_result = []
			# print('all_rec',all_rec)
			if len(all_rec)!=0:
				begin_x = all_rec[0][0]
				end_x = all_rec[-1][0]
				
				if not get_one:
					middle_x = int((begin_x+end_x)/2)
					cv2.line(pred_img,(0,middle_x),(1000,middle_x),(0,0,255),2)
					first_rec = []
					second_rec = []
					for i in range(len(all_rec)):
						x,y,w,h = all_rec[i]
						if x < middle_x:
							first_rec.append(all_rec[i])
						else :
							second_rec.append(all_rec[i])
					cv2.resize(img, (img_ori_to_get_shape[1], img_ori_to_get_shape[0]))
				else :
					first_rec = all_rec
				if len(first_rec) == 0:
					return img_result

				first_rec = self.filterRecList(first_rec)
				
				pt1 ,pt2,return_img_1 = self.findTwo(first_rec,img)
				img_result.append(return_img_1)
				cv2.imwrite('./third.jpg',return_img_1)#*255)
				print('first_rec',first_rec)
				if not get_one:
					if len(second_rec) == 0:
						return img_result
					second_rec = self.filterRecList(second_rec)
					pt3 ,pt4,return_img_2 = self.findTwo(second_rec,img)
					cv2.imwrite('./forth.jpg',return_img_2)#*255)
					cv2.imwrite('./first.jpg',img)
					
					print('second_rec',second_rec)
					img_result.append(return_img_2)
			cv2.imwrite('./second.jpg',pred_img)#*255)
			# return [return_img_1,return_img_2],[[pt1[0],pt1[1],pt1[2]-pt1[0],pt1[3]-pt1[1]],[pt2[0],pt2[1],pt2[2]-pt2[0],pt2[3]-pt2[1]]]
			return img_result

