import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import  matplotlib.pyplot as plt 

import glob
import pandas as pd
import cv2
import os
import math
# use pca to correct image

def showImg(name,img):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name,img)
    cv2.waitKey(2000)
   
path_list =glob.glob('../Archive_to_split_2/*.jpg')
path = '../region/2IMG_20200709_110101.jpg.jpg'#1IMG_20200709_110049.jpg.jpg'#'Archive/correct_2.jpg'#'../hanjun_2.jpg'#
import numpy as np
import matplotlib.pyplot as plt


def isWhite(img,i,j):
    shape = img.shape
    if i >0 and i<shape[0]-1 and j>0 and j<shape[1]-1:
        return img[i-1][j] == 255 and img[i+1][j] == 255 and img[i][j-1] == 255 and img[i][j+1] == 255 and img[i-1][j-1] == 255 and img[i-1][j+1] == 255 and img[i+1][j-1] == 255 and img[i+1][j+1] == 255
    else :
        return False


def getDst(angle,img_color):
    shape = img_color.shape
    dst = img_color#np.zeros([int(shape[0]), int(shape[1]),int(shape[2])], dtype=int)
    # for i in range(shape[0]):
    i  = shape[0]-1
    
    j = 0
    counter = 0
    while i >= 0:
        for j in range(shape[1]):
            r = math.sqrt(i*i+j*j)
            ii = abs(int(r*math.cos(angle)))
            jj = abs(int(r*math.sin(angle)))
            if ii >= 0 and ii < shape[0] and jj >= 0 and jj < shape[1]:
                dst[ii][jj] = img_color[i][j]
        i = i - 1
    return dst

def Hough(img_color):
    # img_color=cv2.flip(img_color,1)
    img_binary = cv2.cvtColor(img_color,cv2.COLOR_RGB2GRAY)
    img_binary = cv2.Canny(img_binary,0,255,apertureSize = 3)
    _,img_binary = cv2.threshold(img_binary, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # showImg('lunkuo',img_binary)   
    shape = img_binary.shape
    
    ratio = float(shape[0])/float(shape[1])
    shape = img_binary.shape
    shape = img_binary.shape

    D = math.sqrt(shape[0]*shape[0]+shape[1]*shape[1])
    angle_step = 0.2
    angle_limit = 90.0/angle_step
    distance_limit = D/0.1
    vote_matrix = np.zeros([int(distance_limit), int(angle_limit)], dtype=int)
    vote_max = 0
    angle_result = 0.0
    i  = shape[0]-1
    
    j = 0
    counter = 0
    while i >= 0:
        while j < shape[1]:
            if img_binary[i][j] == 255:
                ii = shape[0]-i-1
                jj = j
                angle = 0.0
                while angle/angle_step < angle_limit:
                    d = ii*math.sin(math.radians(angle))+ jj*math.cos(math.radians(angle))
                    d = abs(int(round(d,1)/0.1))
                    t = int(angle/angle_step)

                    vote_matrix[d][t] = vote_matrix[d][t]+1
                    if vote_max <= vote_matrix[d][t] and t!=0.0:#and abs(90-angle) >=3 and abs(angle) >=3 :
                        vote_max = vote_matrix[d][t]
                        angle_result = 90-angle
                        if angle < 45:
                            angle_result = -angle
                    angle = angle + angle_step
            j = j+1
        counter =  counter +1
        if counter >= 50:
            break
        j = 0
        i = i-1
    shape = img_color.shape
    matRotate = cv2.getRotationMatrix2D((shape[0]*0.5,shape[1]*0.5),angle_result,1)    # 为方便旋转后图片能完整展示，所以我们将其缩小
    dst = cv2.warpAffine(img_color,matRotate,(shape[1],shape[0]))
    return angle_result#,dst




     
if __name__ == "__main__":
     for path in path_list:
        img_color = cv2.imread(path)
        angle,dst= Hough(img_color)


        print(angle)
        print(img_color.shape,dst.shape)

 
  