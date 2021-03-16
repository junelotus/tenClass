#to use si ti to calcutation the  difference of two image
import numpy as np
import cv2
import os
import os.path
import math

def calTI(image1,image2):
    # frame difference
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        rct, image1 = cv2.threshold(image1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        rct, image2 = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    frame_difference = image2 - image1
    mean_value = np.mean(np.array(frame_difference))
    std_result = 0.0

    # for i in range(frame_difference.shape[0]):
    #     for j in range(frame_difference.shape[1]):
    #         t = frame_difference[i][j] - mean_value
    #         std_result = std_result+t*t
    # print(image2)
    # print(image1)
    # print(frame_difference)
    return np.std(frame_difference),np.sum(frame_difference)#,math.sqrt(std_result)#np.mean(np.array(frame_difference))

    # image_x_1=cv2.Sobel(image1,cv2.CV_64F,1,0,ksize=3)  #X方向Sobel
    # absX_1 = cv2.convertScaleAbs(image_x_1)   # 转回uint8
    # # cv2.imshow("absX",absX)
    # image_y_1 = cv2.Sobel(image1,cv2.CV_64F,0,1,ksize=3)  #Y方向Sobel
    # absY_1 = cv2.convertScaleAbs(image_y_1)
    # dst1 = cv2.addWeighted(absX_1,0.5,absY_1,0.5,0)
 

    # image_x_2 = cv2.Sobel(image2,cv2.CV_64F,1,0,ksize=3)  #X方向Sobel
    # absX_2 = cv2.convertScaleAbs(image_x_2)   # 转回uint8
    # # cv2.imshow("absX",absX)
    # image_y_2 = cv2.Sobel(image2,cv2.CV_64F,0,1,ksize=3)  #Y方向Sobel
    # absY_2 = cv2.convertScaleAbs(image_y_2)
    # dst2 = cv2.addWeighted(absX_2,0.5,absY_2,0.5,0)
image1 = cv2.imread('Archive_to_split_3_162050/91.jpg')
image2 = cv2.imread('Archive_to_split_3_162050/109.jpg')
#(1858.jpg')

print(calTI(image1,image2))
