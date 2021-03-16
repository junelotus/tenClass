import numpy as np
import cv2
import os
import os.path
from pathlib import Path
import shutil

def showImg(name,img):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name,img)
    cv2.waitKey(20000)

def toGetHarrPoint(imgColor):
    imgGery = cv2.cvtColor(imgColor, cv2.COLOR_RGB2GRAY)
    imgGery = np.float32(imgGery)
    #  输入图像必须是 float32 ,最后一个参数在 0.04 到 0.05 之间
    dst = cv2.cornerHarris(imgGery,2,3,0.04)
    
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    imgColor[dst>0.01*dst.max()]=[0,0,255]
    
    showImg('dst',imgColor)
    # rct, imgBinary = cv2.threshold(imgGery, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # showImg("binary",imgBinary)
    # scharrx=cv2.Scharr(imgGery,cv2.CV_64F,1,0)
    # scharry=cv2.Scharr(imgGery,cv2.CV_64F,0,1)
    # scharrx=cv2.convertScaleAbs(scharrx)
    # scharry=cv2.convertScaleAbs(scharry)

    # scharrxy=cv2.addWeighted(scharrx,0.5,scharry,0.5,0)
    # showImg("scharrxy",scharrxy)
imgColor = cv2.imread('switch_1.jpg')
toGetHarrPoint(imgColor)