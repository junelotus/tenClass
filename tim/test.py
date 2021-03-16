import tensorflow as tf
import glob # 获取路径
import tensorflow.keras as keras
from matplotlib import pyplot as plt # 绘图
AUTOTUNE = tf.data.experimental.AUTOTUNE # prefetch 根据CPU个数创建读取线程
import os
import math
import numpy as np
import cv2


import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from utils import Decoder
from losses import CTCLoss
from dataset_factory import DatasetBuilder
from model import build_model

from metrics import WordAccuracy

batch_size  = 1
img_to_del = None


parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--images', type=str, required=True, 
#                     help='Image file or folder path.')
parser.add_argument('-t', '--table_path', type=str, required=True, 
                    help='The path of table file.')
parser.add_argument('-m', '--model', type=str, required=True, 
                    help='The saved model.')
parser.add_argument('-w', '--img_width', type=int, default=0, 
                    help='Image width, this parameter will affect the output '
                         'shape of the model, default is 0')
parser.add_argument('--img_channels', type=int, default=3, 
                    help='0: Use the number of channels in the image, '
                         '1: Grayscale image, 3: RGB image')
args = parser.parse_args()

with open(args.table_path, encoding='UTF-8-sig') as f:
    table = [char.strip() for char in f]
_custom_objects = {
    "CTCLoss" :  CTCLoss,
    "WordAccuracy" : WordAccuracy,
}



model = tf.keras.models.load_model(args.model, custom_objects=_custom_objects)#, compile=False)

decoder = Decoder(table)






def read_jpg_gt(path):
    img = tf.io.read_file(path)# 2.0里面的读取都在io里面
    img = tf.image.decode_jpeg(img,channels = 1)
    return img

def load_img_label(path):
    #print(path)
    img = read_jpg_gt(path)
    img = tf.image.resize(img,(64,64))
    img = tf.cast(img, tf.float32)/127.5 - 1#img/127.5-1
    return img


def read_img_and_preprocess(path, img_width=0, img_height=32):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=args.img_channels)
    if not img_width:
        img_shape = tf.shape(img)
        scale_factor = img_height / img_shape[0]
        img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
        img_width = tf.cast(img_width, tf.int32)
    img = tf.image.resize(img, (img_height, img_width)) / 255.0
    return img
 
def showImg(name,img):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name,img)
    cv2.waitKey(20000)

#showImg('name',cv2.imread('./score_map/1.jpg'))
score_label = glob.glob('./score_map/*jpg')
label_score_img = tf.data.Dataset.from_tensor_slices(score_label)
# label_score_data = label_score_img.map(load_img_label,num_parallel_calls = AUTOTUNE).cache().batch(batch_size)
# for score in label_score_data:
# 	##print(score)
# 	break
def read_jpg(path):
    img = tf.io.read_file(path)# 2.0里面的读取都在io里面
    img = tf.image.decode_jpeg(img,channels = 3)
    return img

def load_img_train(path):
    img = read_jpg(path)
    img = tf.image.resize(img,(512,512)) # resize 会使得在后续显示的时候不是none none
    img = tf.cast(img, tf.float32)/127.5 - 1#img/127.5-1
    return img

# def load_img_train(self):
#         img = tf.image.resize(img_to_del,(512,512)) # resize 会使得在后续显示的时候不是none none
#         img = tf.cast(img, tf.float32)/127.5 - 1 #img/127.5-1
#         return img
def jifentu(img):
	if len(img.shape) == 3:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		rct, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return np.sum(img)
def to_get_begin_end_time(path):
	img_ori = glob.glob(path)#('./1_ch4_training_images/img_1.jpg')#
	batch_size = 1
	train_img = tf.data.Dataset.from_tensor_slices(img_ori)
	batch_size = 1
	train_data = train_img.map(load_img_train,num_parallel_calls = AUTOTUNE).cache().batch(batch_size)#.shuffle(buffer_size)
	
	for t in train_data:
		pred_score,angle_map,geo_map = net(t,training=False)
		#print(geo_map.shape[0])
		size_t =64
		pred_img = np.array(tf.reshape(pred_score,(64,64)))
		shape = geo_map.shape
		t = np.array(tf.reshape(geo_map,shape))
		t = tf.squeeze(t,axis=0)
		t = tf.squeeze(t,axis=-1)
		shape = t.shape[1]
		# #print('t.shape',t[1,1,:].shape)
		# #print('shape',shape)
		list_cor = []

		for i in  range(shape):
			for j in range(shape):
				if  np.sum(np.array(t[i,j,:])) > 0.0:
					# #print('i,j',np.sum(np.array(t[i,j,:])))#np.array(t[i,j,:]))
					position = np.array(t[i,j,:])
					# if len(list_cor) == 0 or (list_cor[-1].all()!=position.all()):
					list_cor.append(position)

		img = np.array(tf.reshape(t[:,:,0],(size_t,size_t)))

		list_cor = list_cor
		# #print(list_cor)
		img_ori_to_get_shape = cv2.imread(path).shape

		img = cv2.imread(path)
	img_ori_to_get_shape = img.shape
	# img = cv2.resize(img, (img_ori_to_get_shape[1], img_ori_to_get_shape[0]))
	##print('pred_img.shape',pred_img.shape)
	pred_img = cv2.resize(pred_img, (img_ori_to_get_shape[1], img_ori_to_get_shape[0]))
	##print('pred_img.shape',pred_img.shape)
	# jifentu
	# pred_img = cv2.resize(pred_img, (1280, 720))*255
	# pred_img = cv2.cvtColor(pred_img.astype(np.int8), cv2.COLOR_RGB2GRAY)
	_, pred_img = cv2.threshold(pred_img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	

	contours, _ = cv2.findContours(pred_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[2], reverse=True)
	i = 0
	for cnt in contours:
		x, y, w, h =  cv2.boundingRect(cnt)
		imgsub = img[max(y-10,0):min(img_ori_to_get_shape[0],y+h+10),max(x-100,0):min(x+w+100,img_ori_to_get_shape[1])]

		# contourssub, _ = cv2.findContours(imgsub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
		# contourssub = sorted(contourssub, key=lambda c: cv2.boundingRect(c)[2], reverse=True)
		imgsub = cv2.bitwise_not(imgsub)
		# imgsub = cv2.cvtColor(imgsub, cv2.COLOR_RGB2GRAY)
		# _, imgsub = cv2.threshold(imgsub.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		#print(imgsub.shape)
		# if  imgsub.shape.any()==0:
		cv2.imencode('.jpg', imgsub)[1].tofile('danmu/'+str(i)+'.jpg')
		# cv2.rectangle(img, pt1=(x-15,y-15), pt2=(x+w+15,y+h+15),color=(0, 0, 255), thickness=2)
		
		i = i+1

	p = Path('danmu/')
	
	if p.is_dir():
		img_paths = p.iterdir()
	else:
		img_paths = [p]
	flag = False
	counter_ha = 0
	counter_qian = 0
	counter_qita = 0
	for img_path in img_paths:
		# #print('img_path',img_path)
		img = read_img_and_preprocess(str(img_path))
		img = tf.expand_dims(img, 0)
		y_pred = model(img)
		# #print(y_pred.shape)
		g_decode = decoder.decode(y_pred, method='greedy')[0]
		import re
		# line="this hdr-biz 123 model server 456"
		pattern= '哈'
		# matchObj = re.match( '前', g_decode)  
		# # b_decode = decoder.decode(y_pred, method='beam_search')[0]
		 
		print(f'Path: {img_path}, greedy: {g_decode}')#, beam search: {b_decode}')
		# print('matchObj',matchObj)
		# if g_decode.startswith("前方高能", beg=0,end=len(g_decode)) or g_decode.startswith("尺度", beg=0,end=len(g_decode)):
		if list(g_decode).count(pattern)>=3:
			counter_ha = counter_ha+1
		if  ('前' in g_decode and '方' in g_decode  and  '高' in g_decode and'能'  in g_decode):
			counter_qian =counter_qian+1

		if  ( '尺' in g_decode and'度'  in g_decode) or( '双' in g_decode and'手'  in g_decode and'打'  in g_decode and'字'  in g_decode) or ('以'  in g_decode and'证'  in g_decode and'清'  in g_decode and'白'  in g_decode) or('调'  in g_decode and'低'  in g_decode and'音'  in g_decode and'量'  in g_decode):
			
			counter_qita =counter_qita+1

		if counter_ha>=3 or counter_qian>=2 or counter_qita >=1:

			print('***********************************************')
			print(f'Path: {img_path}, greedy: {g_decode}')#, beam search: {b_decode}')
			flag = True
			break
	return flag






net = tf.keras.models.load_model('net.h5')
path  = './jpg15/danmu2.jpg'#'./1_ch4_training_images/img_3.jpg'#
path = glob.glob('../mp4/Record_2020-09-30-15-06-10_81ba9ef00f16ba1aa23fbaf3ba390c5c.mp4')
#print('path',path)

 #ffmpeg -f image2 -start_number 8000 -i pinjie3/%d.jpg  -vcodec libx264 -r 30 -b 8000k test3.mp4


def getAll(path):
	for p in path:
		counter = 0
		capture = cv2.VideoCapture(p)
		flag = False
		to_pinjie_par = 0
		frame_exists = True
		while capture.isOpened():
			
			
			if flag == True :
				to_pinjie = 0
				while  to_pinjie < 1000 and frame_exists:
					cv2.imencode('.jpg', img_to_del)[1].tofile('./pinjie/'+str(to_pinjie_par)+'.jpg')
					frame_exists, img_to_del = capture.read()
					img_to_del = np.rot90(img_to_del)
					
					to_pinjie = to_pinjie+1
					to_pinjie_par = to_pinjie_par+1
				counter = 360		
			# 每30帧进行一次检测
			while counter < 360 and  frame_exists: 
				timestamp = int(capture.get(cv2.CAP_PROP_POS_MSEC))
				frame_exists, img_to_del = capture.read()
				if  not frame_exists:
					break
				counter = counter+1
			counter = 0
			if  not frame_exists:
				#print('because end')
				break
			# img_to_del = cv2.imread('danmu1.jpg')
			img_to_del = np.rot90(img_to_del)
			cv2.imencode('.jpg', img_to_del)[1].tofile('./now_jpg/1.jpg')
			flag = to_get_begin_end_time('./now_jpg/1.jpg')

			import shutil
			shutil.rmtree('danmu/')
			os.system("mkdir danmu")	
			# break
os.mkdir('danmu')			
getAll(path)