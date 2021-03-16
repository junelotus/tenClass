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
    cv2.waitKey(500)
   
path_list =glob.glob('../region1/*jpg')
path = '../region/2IMG_20200709_110101.jpg.jpg'#1IMG_20200709_110049.jpg.jpg'#'Archive/correct_2.jpg'#'../hanjun_2.jpg'#
import numpy as np
import matplotlib.pyplot as plt


def isWhite(img,i,j):
    shape = img.shape
    if i >0 and i<shape[0]-1 and j>0 and j<shape[1]-1:
        return img[i-1][j] == 255 and img[i+1][j] == 255 and img[i][j-1] == 255 and img[i][j+1] == 255 and img[i-1][j-1] == 255 and img[i-1][j+1] == 255 and img[i+1][j-1] == 255 and img[i+1][j+1] == 255
    else :
        return False



def Hough(img_color):
    # img_color = cv2.imread(filename)
    img_binary = cv2.cvtColor(img_color,cv2.COLOR_RGB2GRAY)
    img_binary = cv2.Canny(img_binary,0,255,apertureSize = 3)
        
    shape = img_binary.shape
    # print(shape)
    # 裁剪为了使当前的数据更少尽量去除边框的影响
    # img_binary = img_binary[50:shape[0]-50,50:shape[1]-50]
    
    ratio = float(shape[0])/float(shape[1])
    # size = (int(shape[0]/2.0), int(shape[1]/2.0))#/ratio))
    # img_binary = cv2.resize(img_binary, size, interpolation=cv2.INTER_LINEAR)
    shape = img_binary.shape
    # print(shape)
    
    #缩放
    # ratio = float(shape[0])/float(shape[1])
    # size = (int(shape[0]/4.0), int(shape[1]/4.0/ratio))
    # for i in range(1):
    #     img_binary = cv2.pyrDown(img_binary)
    shape = img_binary.shape
    # showImg('binary',img_binary)

    D = math.sqrt(shape[0]*shape[0]+shape[1]*shape[1])
    # print(D)
    angle_step = 0.2
    angle_limit = 90.0/angle_step
    distance_limit = D/0.1
    vote_matrix = np.zeros([int(distance_limit), int(angle_limit)], dtype=int)
    vote_max = 0
    angle_result = 0.0

    # i  = shape[0]-1
    
    # j = 0
    # while i >= 0:
    #     while j < shape[1]:
    #         if img_binary[i][j] == 255:
    #             ii = shape[0]-i-1
    #             jj = j
    #             angle = 0.0
    #             while angle/angle_step < angle_limit:
    #                 d = ii*math.sin(math.radians(angle))+ jj*math.cos(math.radians(angle))
    #                 d = abs(int(round(d,1)/0.1))
    #                 t = int(angle/angle_step)

    #                 vote_matrix[d][t] = vote_matrix[d][t]+1
    #                 # print(vote_max,d,t,vote_matrix[d][t],angle)
                    # # print(i,j,angle)
                    # if vote_max < vote_matrix[d][t] and t!=0.0:
                    #     # print('dayu')

                    #     # print(vote_max,d,t,vote_matrix[d][t],angle)
                    #     vote_max = vote_matrix[d][t]
                    #     # print(angle)
                    #     angle_result = 90-angle
                    #     if angle < 45:
                    #         angle_result = -angle
        #                 # print(angle_result)
        #                 # print('\n')
        #                  #- 90.0
        #             angle = angle + angle_step
        #         # i = i-1
        #     j = j+1
        # j = 0
        # i = i-1


    # vote_matrix = np.zeros([int(distance_limit), int(angle_limit)], dtype=int)
   
    # i  = shape[0]-1
    
    # j = 0
    # while i >= 0:
    #     while j < shape[1]:
    #         if img_binary[i][j] == 255:
    #             ii = shape[0]-i-1
    #             jj = j
    #             angle = -90.0
    #             while angle/angle_step < 0:
    #                 d = ii*math.sin(math.radians(angle))+ jj*math.cos(math.radians(angle))
    #                 d = abs(int(round(d,1)/0.1))
    #                 t = int((angle+90)/angle_step)
    #                 vote_matrix[d][t] = vote_matrix[d][t]+1
    #                 # print(vote_max,d,t,vote_matrix[d][t],angle)
    #                 # print(i,j,angle)
    #                 if vote_max < vote_matrix[d][t] and t!=0.0:
    #                     print('dayu')

    #                     print(vote_max,d,t,vote_matrix[d][t],angle)
    #                     vote_max = vote_matrix[d][t]
    #                     angle_result = angle #- 90.0
    #                 angle = angle + angle_step
    #             # i = i-1
    #         j = j+1
    #     j = 0
    #     i = i-1





    for i in range(shape[0]):
        for j in range(shape[1]):
            if img_binary[i][j] == 255:
                angle  = 0.0
                while angle/0.2 < angle_limit:

                    d = j*math.sin(math.radians(angle))+ i*math.cos(math.radians(angle))

                    d = abs(int(round(d,1)/0.1))
                    t = int(angle/0.2)
                    # print(angle)
                    # print(t)
                    vote_matrix[d][t] = vote_matrix[d][t]+1
                    # print(vote_max,d,t,vote_matrix[d][t],angle)

                    if vote_max < vote_matrix[d][t]and t != 0.0:
                        # print('dayu')

                        # print(vote_max,d,t,vote_matrix[d][t],angle)
                        vote_max = vote_matrix[d][t]
                        # angle_result = -angle# - 90.0
                        angle_result = 90-angle
                        if angle < 45:
                            angle_result = -angle
                    angle = angle + 0.2
            # if j > 2:
            #     return 0        
    # for i in range(int(distance_limit)):
    #     for j in range(int(angle_limit)):
    #         if vote_max<vote_matrix[i][j]:
                
    #                 vote_max = vote_matrix[i][j]
    #                 angle_result = float(j)*0.2-90.0
    #                 print(angle_result)
    






   

   
    shape = img_color.shape

    # angle_result = 0-angle_result
    matRotate = cv2.getRotationMatrix2D((shape[0]*0.5,shape[1]*0.5),angle_result,0.5)    # 为方便旋转后图片能完整展示，所以我们将其缩小
    dst = cv2.warpAffine(img_color,matRotate,(shape[0],shape[1]))
    # showImg('dst',dst)
    return angle_result




     
if __name__ == "__main__":
     for path in path_list:
        img_color = cv2.imread(path)
        angle = Hough(img_color)
        print(angle)

# img = cv2.imread(path)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,0,255,apertureSize = 3)
# # showImg('edges',edges)
# lines = cv2.HoughLinesP(edges,1,np.pi/180,200)
# print(lines)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
# cv2.imwrite('houghlines3.jpg',img)
 
  