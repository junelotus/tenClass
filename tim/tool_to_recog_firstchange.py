import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import  matplotlib.pyplot as plt 
import math

import glob
import pandas as pd
import cv2
import os
from test_deblur import DeBlur
from num_classification.test_classification  import Classification
def takeSecond(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return x
def takeY(cnt):
     x, y, w, h = cv2.boundingRect(cnt)
     return y

def showImg(name,img):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name,img)
    cv2.waitKey(1000)

def getModelJEPGAnd15Positions(path):
    imageModel =cv2.imread( path)#'model_8.jpg')
    imageModel = cv2.cvtColor(imageModel,cv2.COLOR_RGB2GRAY)
    rct,imageModel = cv2.threshold(imageModel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours_model , _ = cv2.findContours ( imageModel ,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_model.sort(key=takeSecond,reverse=False)
    contours_model.sort(key=takeY,reverse=False)
    contours_model_result =[]
    for cnt in contours_model:
                x1,y1,w1,h1 = cv2.boundingRect(cnt)
                print('area')
                aear = w1*h1
                print(aear)
                img_curent_sub = imageModel[y1:y1+h1,x1:x1+w1]
                
                
                # print("heeee")
                # print(L2Loss(img_curent_sub,model))
                shape_sub = img_curent_sub.shape
                if not (shape_sub[0]<30 or  shape_sub[1]<20 or shape_sub[0]>=0.9*imageModel.shape[0] or shape_sub[1]>=0.9*imageModel.shape[1]) :
                    contours_model_result.append(cnt)
                    # showImg("sub",img_curent_sub)
    imageModel = imageModel/127.5-1
    return  imageModel,contours_model_result



def L2Loss(img,model):
    if len(img.shape) == 3:
    
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        rct,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if len(model.shape) == 3:
        model = cv2.cvtColor(model,cv2.COLOR_RGB2GRAY)
        rct,model = cv2.threshold(model,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # minx = min(img.shape[0],model.shape[0])
    # miny = min(img.shape[1],model.shape[1])
    shape = model.shape
    size = (shape[1], shape[0])
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)  
    result = 0.0
    i = 0
    j = 0
    # cv2.namedWindow('img***',cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('img***',img)
    # cv2.namedWindow('model***',cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('model***',model)

    # print(img)
   
    # print(model)
    d = 0
    for  i in range(shape[0]):
        for j in range(shape[1]):
            t = model[i][j] - img[i][j]
            # print(t)
            result = result+ t*t
            # d = d+1
            # print(i,j)
            # j = 0
        # print('d=')
        # print(d)
    # print(shape)
    return result


def showImgByCnt(name,img,listsOfCnt):
        for x1,y1,w1,h1,  L2Loss_result in listsOfCnt:   # i in range(1):#    
                cv2.namedWindow('ori_'+name,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
                cv2.imshow('ori_'+name,img)
                cv2.namedWindow('current_'+name,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
                cv2.imshow('current_'+name,img[y1:y1+h1,x1:x1+w1])
                print(L2Loss_result)
                # cv2.waitKey(1000)
    

def findROI(image_color,imageModel,w_,h_,l2_loss_1,l2_loss_2,ratio):
    # img_color  = cv2.imread(p)
    img = cv2.cvtColor(image_color,cv2.COLOR_RGB2GRAY)
    rct,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours , hierarchy = cv2.findContours ( img ,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE ) cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    shape =img.shape
    listsOfCnt = []
    x_before = -1
    y_before = -1
    w_before = -1
    h_before = -1
    contours.sort(key=takeSecond,reverse=False)
    # print("dddd")
    # print(len(contours))
    # print(x_before,y_before,w_before,h_before)
    # print(path)
    for cnt in contours:
            x1,y1,w1,h1 = cv2.boundingRect(cnt)
            if w1*h1<w1<w_*h_ or w1<w_ or h1<h_ :#or  w1*h1 >shape[0]*shape[1]/15.0:# or w1 >shape[0]/3 or h1>shape[1]/ 5 :
                continue
            # print("eeeee")
            imgCurrent = img[y1:y1+h1,x1:x1+w1]  

            imgCurrent_color = image_color[y1:y1+h1,x1:x1+w1]
            imgCurrent = imgCurrent/127.5-1
            L2Loss_result =  L2Loss(imgCurrent,imageModel)
            # print(L2Loss_result)
            
            # print((L2Loss_result))
            # if not(L2Loss_result > 900000 and  L2Loss_result <1100000) or float(imgCurrent.shape[0])/float(imgCurrent.shape[1]) <1:
            #     continue

            # to use
            # if not(L2Loss_result > 1100000 and  L2Loss_result < 1350000) or float(imgCurrent.shape[0])/float(imgCurrent.shape[1]) <1:
            #     continue
            #50000 140000
            if not(L2Loss_result > l2_loss_1 and  L2Loss_result < l2_loss_2) or float(imgCurrent.shape[0])/float(imgCurrent.shape[1]) < ratio:
                continue
            # print('lallal')
            # print(x_before)
            if x_before != -1:
                # if x1>=x_before  and y1>=y_before and x1+w1<=x_before+w_before and y1+h1 <= y_before+h_before:
                while len(listsOfCnt)>0 and (x1>=listsOfCnt[-1][0]  and y1>=listsOfCnt[-1][1] and x1+w1<=listsOfCnt[-1][0]+listsOfCnt[-1][2] and y1+h1 <= listsOfCnt[-1][1]+listsOfCnt[-1][3]):
                    # print('ttttttt')
                    
                    # print()
                    # print(y)
                    del(listsOfCnt[-1])
            if len(listsOfCnt) == 0:
                    x_before = -1
                    y_before = -1
                    w_before = -1
                    h_before = -1
                    # continue
            # print(x1,x_before)
            # print(y1,y_before)
            # print(x1+w1,x_before+w_before)
            # print(y1+h1,y_before+h_before)
            # print('append')
            listsOfCnt.append([x1,y1,w1,h1, L2Loss_result])


            x_before = x1
            y_before = y1
            w_before = w1
            h_before = h1
            #  if not(L2Loss_result > 240 and  L2Loss_result < 310) or float(imgCurrent.shape[0])/float(imgCurrent.shape[1]) <1:
            #     continue
    #         # 1229583
    # print(len(listsOfCnt))
    # cv2.namedWindow('model',cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('model',imageModel)
    # cv2.namedWindow('ori',cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('ori',img_color)
    # i = 0
    

    return listsOfCnt
# cv2.imwrite("model_2.jpg",imageModel)

def reShape(imageModel,imgCuerrnt):
        shape = imageModel.shape
        size = (shape[1], shape[0])
        imgCuerrnt = cv2.resize(imgCuerrnt, size, interpolation=cv2.INTER_LINEAR)
        return imgCuerrnt
def getRowWhiteCount(img,beginx,beginy,w,h):
    j = beginx
    counter = 0 
    while j< min(beginx+h,img.shape[1]-1):
        if img[beginy][j] == 255:
            counter= counter+1
        j = j+1
    return counter

def getLieWhiteCount(img,beginx,beginy,w,h):
    i = beginy
    counter = 0 
    while i<min(beginy+w,img.shape[0]-1):
        if img[i][beginx] == 255:
            counter= counter+1
        i = i+1
    return counter

def slideWindow(img,beginx,beginy,endx,endy,w,h,step):#36 54
    shape = img.shape
    oriy = beginy
    result_list = []
    beginx = max(0,beginx)
    beginy = max(0,beginy)
    print(beginx,endx,beginy,endy)
    while beginx < endx:
        while beginy < endy:
            counter1 = getRowWhiteCount(img,beginx,beginy,w,h)
            counter2 = getRowWhiteCount(img,beginx,min(shape[0]-1,beginy+h),w,h)
            counter3 = getLieWhiteCount(img,beginx,beginy,w,h)
            counter4 = getLieWhiteCount(img,min(shape[1]-1,beginx+w),beginy,w,h)
            # print("counter")
            # print(counter1,counter2,counter3,counter4)
            # showImg("slidehehe",img[beginy:beginy+h,beginx:beginx+w])
            
            counter = counter1+counter2+counter3+counter4
            # print(counter1)
            # print(counter2)
            # print(counter3)
            # print(counter4)
            # print(counter)
            if counter ==  (2.0*(w+h)) :
                print("shimashmashima")
                result_list.append([beginx,beginy,w,h])
                break
            
            beginy = beginy+step
        if len(result_list) > 0:
                break
        beginy = oriy
        beginx = beginx+step
    return result_list


def getAvaiableImg(p,listsOfCnt,w,h,step,imageModel):
    listsOfImg = []
    listsOfCnts = []
    img_color = cv2.imread(p)
    img = cv2.cvtColor(img_color,cv2.COLOR_RGB2GRAY)
    rct,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print('test3')
    print(len(listsOfCnt))
    for x1,y1,w1,h1, L2Loss_result in listsOfCnt:
        # print('test1')
        # listsOfCntCurrent = []
        imgCuerrnt = img[y1:y1+h1,x1:x1+w1]
        imgCuerrnt = reShape(imageModel,imgCuerrnt)
        # result_list = slideWindow(imgCuerrnt,0,0,w,h,step)
        # if len(result_list) < 15:
        #     continue
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        imgCuerrnt = cv2.dilate(imgCuerrnt, kernel)
        imgCuerrnt = cv2.erode(imgCuerrnt, kernel)
        contours , hierarchy = cv2.findContours ( imgCuerrnt ,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE ) cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
        contours.sort(key=takeSecond,reverse=False)
        contours.sort(key=takeY,reverse=False)
        # print(len(contours))
        if len(contours) < 15:
            continue
        listsOfImg.append(reShape(imageModel,img_color[y1:y1+h1,x1:x1+w1]))
        listsOfCnts.append(contours)
    return listsOfImg,listsOfCnts



# print(imageModel.shape)
def setUtils(esc_encoder_wokl_2_path,ebc_encoder_wokl_2_path,eb_encoder_wokl_2_path,gen_s_wokl_2_path,tenClass_path):
    deblur_tool = DeBlur(1,esc_encoder_wokl_2_path,ebc_encoder_wokl_2_path,eb_encoder_wokl_2_path,gen_s_wokl_2_path)
    deblur_tool.set_buffer_size(1)
    class_tool = Classification(tenClass_path)
    class_tool.set_batch_size(15)
    return deblur_tool,class_tool
# path = glob.glob('./jpg15/*.jpg')#./Archive/*.jpg')#./Archive/*.jpg')#./jpg10/IMG_20200622_110414.jpg')#
# i = 0.0
# j = 0.0
# count_1 = 0
# count_3 = 0
# model = cv2.imread('1_model_8.jpg')
# model = cv2.cvtColor(model,cv2.COLOR_RGB2GRAY)
# rct,model = cv2.threshold(model,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# list_path = []
# #IMG_20200706_142216.jpg 
# list_path.append('./jpg17/*.jpg')
# list_path.append('./jpg5/*.jpg')
# list_path.append('./jpg6/*.jpg')
# list_path.append('./jpg7/*.jpg')
# list_path.append('./jpg8/*.jpg')
# list_path.append('./jpg9/*.jpg')
# list_path.append('./jpg10/*.jpg')
# list_path.append('./jpg11/*.jpg')
# list_path.append('./jpg12/*.jpg')
# list_path.append('./jpg13/*.jpg')
def getTimeresult(path,deblur_tool,class_tool,contours_model_result):
        # for path_ in list_path:
        #     print(path_)
        #     path = glob.glob(path_)
        #     for p in path:
        #         seconfOfPath  = p.split('/')[2]
                th = 0
                seconfOfPath  = path.split('/')[2]
                img_color=cv2.imread(path)

                listsOfCnt = findROI(img_color,imageModel,300,300,50000,140000,1)

                # print('test2')
                # print(len(listsOfCnt))
                listsOfImg,listsOfCnts = getAvaiableImg(path,listsOfCnt,54,36,5,imageModel)
                # if len(listsOfImg)<3:
                #     continue
                namee  = 1
                for img in listsOfImg:
                    # pathToSave = 'region/'+str(namee)+seconfOfPath+'.jpg'
                    # namee  = namee+1
                    # cv2.imencode('.jpg', img)[1].tofile(pathToSave)
                    result_list_fiften = []
                    a = 'a'
                    img = reShape(imageModel,img)
                    img_gery = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                    rct,img_binary = cv2.threshold(img_gery,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    showImg('ori',img)
                    
                    for cnt in contours_model_result:
                            x1,y1,w1,h1 = cv2.boundingRect(cnt)
                            
                            result_final = slideWindow(img_binary,x1-10,y1-10,x1+10,y1+10,w1+5,h1+5,1)
                            print(result_final)
                            print(x1,y1,w1,h1)
                            img_curent_sub1 = img[y1:y1+h1,x1:x1+w1]
                            # showImg("sub1",img_curent_sub1)
                            if len(result_final) > 0:
                                x1 = result_final[0][0]
                                y1 = result_final[0][1]
                                w1 = result_final[0][2]
                                h1 = result_final[0][3]
                            print(x1,y1,w1,h1)
                            print('area')
                            aear = w1*h1
                            print(aear)
                            img_curent_sub = img[y1:y1+h1,x1:x1+w1]
                            shape_sub = img_curent_sub.shape
                            if not (shape_sub[0]<30 or  shape_sub[1]<20 or shape_sub[0]>=0.9*img.shape[0] or shape_sub[1]>=0.9*img.shape[1]) :
                                # showImg("sub",img_curent_sub)
                                pathToSave = './Archive_to_split_1/'+str(th)+'_'+seconfOfPath
                                cv2.imencode('.jpg', img_curent_sub)[1].tofile(pathToSave)
                                deblur_tool.set_path_to_deblur(pathToSave)
                                deblur_tool.set_dateset()
                                deblured_image = deblur_tool.to_get_result()
                                
                                
                                pathToSave = './Archive_to_split_1/'+a+'_'+str(th)+'_'+seconfOfPath
                                print(pathToSave)
                                img_gery = np.array(deblured_image[0]*0.5+0.5)*255
                                cv2.imencode('.jpg', img_gery)[1].tofile(pathToSave)
                                th = th+1
                                
                                class_tool.set_path(pathToSave)#'./Archive_to_split'+a+'*.jpg')
                    
                                result_list,list_confidence = class_tool.print_out()
                                result_list_fiften.append(result_list[0])
                    print('***********************')
                    print(result_list_fiften)
                print(listsOfCnt)
                    # a = str((int(a)+1))
                return listsOfCnt,result_list,list_confidence

# deblur_tool,class_tool = setUtils('./cycleGAN_to_deblur_train/esc_encoder_wokl_3.h5',\
#                     './cycleGAN_to_deblur_train/ebc_encoder_wokl_3.h5',
#                     './cycleGAN_to_deblur_train/eb_encoder_wokl_3.h5',
#                     './cycleGAN_to_deblur_train/gen_s_wokl_3.h5',
#                     './ocr_model_train/tenClass_cpu.h5')
# imageModel,contours_model_result = getModelJEPGAnd15Positions("model_8.jpg")
# getTimeresult('./jpg15/IMG_20200706_142152.jpg',deblur_tool,class_tool,contours_model_result)