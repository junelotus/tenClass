
import numpy as np
import cv2
import os
import os.path
from pathlib import Path
import shutil
import datetime
import testbk
from skimage.measure import  compare_ssim
import time 

img = cv2.imread('/Users/xhj/opencv_bitbucket/timestampocr/result3/img50.jpg') 
def showImg(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, img)
    cv2.waitKey(1000000)
print(img.shape)
showImg('50',img)

shape[0] 下
shape[1] 右
第一个坐标点是向右的
第二个坐标点是向下的