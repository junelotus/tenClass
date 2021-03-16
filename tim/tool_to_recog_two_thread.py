# -*- coding: utf-8 -*-


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
import threading

from test_deblur import DeBlur
from num_classification.test_classification import Classification
from tool_perspectiv import *

python_path = os.environ["PYTHONPATH"].split(':')[0]


# Compators
def takeSecond(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return x


def takeY(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return y


# tool func to show image
def showImg(name, img,time=1000):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, img)
    cv2.waitKey(time)


# according to the model img to get positions of every number
# path: the path of the image will be cal
def getModelJEPGAnd15Positions(path):
    imageModel = cv2.imread(path)
    imageModel = cv2.cvtColor(imageModel, cv2.COLOR_RGB2GRAY)
    rct, imageModel = cv2.threshold(imageModel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_model, _ = cv2.findContours(imageModel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_model.sort(key=takeSecond, reverse=False)
    contours_model.sort(key=takeY, reverse=False)
    contours_model_result = []
    for cnt in contours_model:
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        # print('area')
        aear = w1 * h1
        # print(aear)
        img_curent_sub = imageModel[y1:y1 + h1, x1:x1 + w1]
        shape_sub = img_curent_sub.shape
        if not (shape_sub[0] < 30 or shape_sub[1] < 20 or shape_sub[0] >= 0.9 * imageModel.shape[0] or
                shape_sub[1] >= 0.9 * imageModel.shape[1]):
            contours_model_result.append(cnt)
    imageModel = imageModel / 127.5 - 1
    return imageModel, contours_model_result


# frame difference to compare two image
# then set threshold to judge the degree of change
def L2Loss(img, model):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rct, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if len(model.shape) == 3:
        model = cv2.cvtColor(model, cv2.COLOR_RGB2GRAY)
        rct, model = cv2.threshold(model, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    shape = model.shape
    size = (shape[1], shape[0])
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return np.sum(np.square(img - model))

# def calculate(image1, image2):
#     counter_1 = 0
#     counter_2 = 0
#     print(image1)
#     for i in range(image1.shape[0]):
#         for j in range(image1.shape[1]):
#             if image1[j][i] !=  0:
#                 counter_1 = counter_1+1
#             if image2[j][i] !=  0:
#                 counter_2 = counter_2+1
#     return np.abs(counter_1-counter_2)

    # hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    # hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # # 计算直方图的重合度
    # degree = 0
    # for i in range(len(hist1)):
    #     if hist1[i] != hist2[i]:
    #         degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
    #     else:
    #         degree = degree + 1
    # degree = degree / len(hist1)
    # return degree
def L2Loss2(img,model,frame_end_time=0):
    # img = img[15:img.shape[0]-15,15:img.shape[1]-15]
    # model = model[15:model.shape[0]-15,15:model.shape[1]-15]
    # img = img[0: img.shape[0],int(img.shape[1]/5*3):img.shape[1]]
    # model = model[0: model.shape[0],int(model.shape[1]/5*3):model.shape[1]]
    # if frame_end_time == 23916:
    #     showImg('img',img)
    #     showImg('model',model)
        # cv2.imencode('.jpg', img)[1].tofile('imgjune_1.jpg')
        # cv2.imencode('.jpg', model)[1].tofile('modeljune_1.jpg')
    if len(img.shape) == 3:
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2GRAY)
        rct, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if len(model.shape) == 3:
        model = cv2.cvtColor(np.uint8(model), cv2.COLOR_RGB2GRAY)
        rct, model = cv2.threshold(model, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    shape = model.shape
    img_list = []
    model_list = []
    img_1 = img[0:int(shape[0]/3),0:shape[1]]

    img_list.append(img_1)
    img_2 = img[int(shape[0]/3):int(shape[0]/3*2),0:shape[1]]
    img_list.append(img_2)
    img_3 = img[int(shape[0]/3*2):int(shape[0]),0:shape[1]]
    img_list.append(img_3)
    
    model_1 = model[0:int(shape[0]/3),0:shape[1]]
    model_2 = model[int(shape[0]/3):int(shape[0]/3*2),0:shape[1]]
    model_3 = model[int(shape[0]/3*2):int(shape[0]),0:shape[1]]
    # showImg('model_1',model_1)
    model_list.append(model_1)
    model_list.append(model_2)
    model_list.append(model_3)
    couter_chazhi = []
    for i in range(3):
        counter_1 = 0
        counter_2 = 0
        img = img_list[i]
        img =np.uint8(img)
        model = model_list[i]
        model = np.uint8(model)
        couter_chazhi.append(np.sum(np.square(img - model)))#(np.abs(counter_1-counter_2))
        
    return couter_chazhi
def L2Loss1(img, model):
    # print('L2Loss1',img.shape,img.ndim,model.shape,model.ndim)

    # print(img.shape)
    # print(model.shape)
    # img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    # model = cv2.cvtColor(model.astype(np.float32), cv2.COLOR_RGB2GRAY)
            
    # hist_cv = cv2.calcHist([img],[0],None,[256],[0,256])
    # cv2.threshold(model, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # if len(img.shape) == 3:
    #     img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2GRAY)
    #     rct, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # if len(model.shape) == 3:
    #     model = cv2.cvtColor(np.uint8(model), cv2.COLOR_RGB2GRAY)
    #     rct, model = cv2.threshold(model, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # img = img[15:img.shape[0]-15,15:img.shape[1]-15]
    # # model = model[15:model.shape[0]-15,15:model.shape[1]-15]
    # # img = img[0: img.shape[0],int(img.shape[1]/5*3):img.shape[1]]
    # # model = model[0: model.shape[0],int(model.shape[1]/5*3):model.shape[1]]
    # if len(img.shape)==2:
    #     img = np.uint8(img)#float32(img)
    #     # rct, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # if len(model.shape)==2:
    #     model = np.uint8(model)#float32(model)
        # rct, model = cv
    # showImg('img',img)
    # showImg('model',model)
    #orb
    return compare_ssim(img,model,multichannel=True)
    shape = model.shape
    img_list = []
    model_list = []
    img_1 = img[0:int(shape[0]/3),0:shape[1]]
    img_list.append(img_1)
    img_2 = img[int(shape[0]/3):int(shape[0]/3*2),0:shape[1]]
    img_list.append(img_2)
    img_3 = img[int(shape[0]/3*2):int(shape[0]),0:shape[1]]
    img_list.append(img_3)
    
    model_1 = model[0:int(shape[0]/3),0:shape[1]]
    model_2 = model[int(shape[0]/3):int(shape[0]/3*2),0:shape[1]]
    model_3 = model[int(shape[0]/3*2):int(shape[0]),0:shape[1]]
    model_list.append(model_1)
    model_list.append(model_2)
    model_list.append(model_3)
    similary_sum = 1.0
    similary_list = []
    for i in range(3):
    
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img_list[i], None)
        kp2, des2 = orb.detectAndCompute(model_list[i], None)

            # 提取并计算特征点
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

            # knn筛选结果
        matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)

            # 查看最大匹配点数目
        good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
        # print(len(good))
        # print(len(matches))
        similary = len(good) / len(matches)
        similary_list.append(similary)
        # print("两张图片相似度为:%s" % similary)
        similary_sum = similary_sum*similary
        # print("两张图片相似度为:%s" % similary)
    return similary_list
    #orb
    #hist
    # shape = model.shape
    # size = (shape[1], shape[0])
    # img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    
    
    # img_1 = img[0:int(shape[0]/3),0:shape[1]]
    # # showImg('img_1',img_1)
    # img_2 = img[int(shape[0]/3):int(shape[0]/3*2),0:shape[1]]
    # img_3 = img[int(shape[0]/3*2):int(shape[0]),0:shape[1]]
    # hist_img_1 = cv2.calcHist([img_1],[0],None,[100],[0,100])
    
    # hist_img_2 = cv2.calcHist([img_2],[0],None,[100],[0,100])
    # hist_img_3 = cv2.calcHist([img_3],[0],None,[100],[0,100])
    
    # model_1 = model[0:int(shape[0]/3),0:shape[1]]
    # # showImg('model_1',model_1)
    # model_2 = model[int(shape[0]/3):int(shape[0]/3*2),0:shape[1]]
    # model_3 = model[int(shape[0]/3*2):int(shape[0]),0:shape[1]]
    # hist_model_1 = cv2.calcHist([model_1],[0],None,[100],[0,100])
    # hist_model_2 = cv2.calcHist([model_2],[0],None,[100],[0,100])
    # hist_model_3 = cv2.calcHist([model_3],[0],None,[100],[0,100])
    # # print(hist_img)
    # # print(hist_model)
    # return np.abs(np.sum(hist_img_1+hist_img_2+hist_img_3)-np.sum(hist_model_1+hist_model_2+hist_model_3))
    #hist
    shape = model.shape
    img_list = []
    model_list = []
    img_1 = img[0:int(shape[0]/3),0:shape[1]]
    # print(img_1)
    # showImg('img_1',img_1)
    img_list.append(img_1)
    img_2 = img[int(shape[0]/3):int(shape[0]/3*2),0:shape[1]]
    img_list.append(img_2)
    img_3 = img[int(shape[0]/3*2):int(shape[0]),0:shape[1]]
    img_list.append(img_3)
    
    model_1 = model[0:int(shape[0]/3),0:shape[1]]
    model_2 = model[int(shape[0]/3):int(shape[0]/3*2),0:shape[1]]
    model_3 = model[int(shape[0]/3*2):int(shape[0]),0:shape[1]]
    # showImg('model_1',model_1)
    model_list.append(model_1)
    model_list.append(model_2)
    model_list.append(model_3)
    
    # print(img)
    couter_chazhi = []
    for i in range(3):
        counter_1 = 0
        counter_2 = 0
        img = img_list[i]
        img =np.uint8(img)
        model = model_list[i]
        model =np.uint8(model)
        # # print(img)
        # counter_1 = np.bincount(np.array(img).flatten())[255]
        # counter_2 = np.bincount(np.array(model).flatten())[255]
        # # for i in range(img.shape[0]):
        # #     for j in range(img.shape[1]):
        # #         if img[i][j] !=  0:
        # #             counter_1 = counter_1+1
        # #         if model[i][j] !=  0:
        # #             counter_2 = counter_2+1
        couter_chazhi.append(np.sum(np.square(img - model)))#(np.abs(counter_1-counter_2))
        # print('chazhi',np.abs(counter_1-counter_2))
    return couter_chazhi#np.abs(counter_1-counter_2)#np.abs()
# tool func to show image
def showImgByCnt(name, img, listsOfCnt):
    for x1, y1, w1, h1, L2Loss_result in listsOfCnt:
        cv2.namedWindow('ori_' + name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('ori_' + name, img)
        cv2.namedWindow('current_' + name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('current_' + name, img[y1:y1 + h1, x1:x1 + w1])
        # print(L2Loss_result)
        # cv2.waitKey(1000)


# to find ROI in an color image
# image_color: the color image to find ROI
# imageModel: according to this image to find ROI,the region more similar the better
# w_: threshold of ROI wight
# h_:threshold of ROI height
# l2_loss_1:lower limmit of loss
# l2_loss_2:upper limmit of loss
# ratio : threshold of wight/height
def findROI(image_color, imageModel, w_, h_, l2_loss_1, l2_loss_2, ratio):
    # img_color  = cv2.imread(p)
    img = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
    rct, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    shape = img.shape
    listsOfCnt = []
    listsOfCntNormalOrder = []
    x_before = -1
    y_before = -1
    w_before = -1
    h_before = -1
    contours.sort(key=takeSecond, reverse=False)
    for cnt in contours:
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        if w1 * h1 < w1 < w_ * h_ or w1 < w_ or h1 < h_:
            continue
        imgCurrent = img[y1:y1 + h1, x1:x1 + w1]
        imgCurrent = imgCurrent / 127.5 - 1
        L2Loss_result = L2Loss(imgCurrent, imageModel)
        if not ((L2Loss_result > l2_loss_1) and (L2Loss_result < l2_loss_2)) or \
                float(imgCurrent.shape[0]) / float(imgCurrent.shape[1]) < ratio:
            continue
        if x_before != -1:
            while len(listsOfCnt) > 0 and (x1 >= listsOfCnt[-1][2] and y1 >= listsOfCnt[-1][3] and
                                           x1 + w1 <= listsOfCnt[-1][2] + listsOfCnt[-1][0] and y1 + h1 <=
                                           listsOfCnt[-1][3] + listsOfCnt[-1][1]):
                del (listsOfCnt[-1])
                del (listsOfCntNormalOrder[-1])
        if len(listsOfCnt) == 0:
            x_before = -1
            y_before = -1
            w_before = -1
            h_before = -1
            # continue
        listsOfCnt.append([w1, h1, x1, y1, L2Loss_result])
        listsOfCntNormalOrder.append(cnt)
        x_before = x1
        y_before = y1
        w_before = w1
        h_before = h1
    return listsOfCnt, listsOfCntNormalOrder


# imgCuerrnt: the image to reshape
# imageModel :the model of  reshape
def reShape(imageModel, imgCuerrnt):
    shape = imageModel.shape
    size = (shape[1], shape[0])
    imgCuerrnt = cv2.resize(imgCuerrnt, size, interpolation=cv2.INTER_LINEAR)
    return imgCuerrnt


# get the white pixel(foreground) of image's row
# img: the image to get white pixel
# beginx : the x begin of region
# beginy :the x begin of region
# w : the wight  of region
# h : the height of region
def getRowWhiteCount(img, beginx, beginy, w, h):
    j = beginx
    counter = 0
    while j < min(beginx + h, img.shape[1] - 1):
        if img[beginy][j] == 255:
            counter = counter + 1
        j = j + 1
    return counter


# get the white pixel(foreground) of image's lie
# img: the image to get white pixel
# beginx : the x begin of region
# beginy :the x begin of region
# w : the wight  of region
# h : the height of region
def getLieWhiteCount(img, beginx, beginy, w, h):
    i = beginy
    counter = 0
    while i < min(beginy + w, img.shape[0] - 1):
        if img[i][beginx] == 255:
            counter = counter + 1
        i = i + 1
    return counter


# slide window to get the every nums' exactly position
def slideWindow(img, beginx, beginy, endx, endy, w, h, step):
    shape = img.shape
    oriy = beginy
    result_list = []
    beginx = max(0, beginx)
    beginy = max(0, beginy)
    # print(beginx, endx, beginy, endy)
    while beginx < endx:
        while beginy < endy:
            counter1 = getRowWhiteCount(img, beginx, beginy, w, h)
            counter2 = getRowWhiteCount(img, beginx, min(shape[0] - 1, beginy + h), w, h)
            counter3 = getLieWhiteCount(img, beginx, beginy, w, h)
            counter4 = getLieWhiteCount(img, min(shape[1] - 1, beginx + w), beginy, w, h)

            counter = counter1 + counter2 + counter3 + counter4
            if counter == (2.0 * (w + h)):
                # print("shimashmashima")
                result_list.append([beginx, beginy, w, h])
                break

            beginy = beginy + step
        if len(result_list) > 0:
            break
        beginy = oriy
        beginx = beginx + step
    return result_list


# get the corrected image
def getAvaiableImgUseCorner(img_color, listsOfCnt, w, h, step, imageModel, flag_of_path):
    # print('getAvaiableImgUseCorner')
    listsOfImg = []
    # listsOfCnts = []
    img = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    rct, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    shape = img_color.shape

    for w1, h1, x1, y1, L2Loss_result in listsOfCnt:
        imgCuerrnt = img[y1:y1 + h1, x1:x1 + w1]
        imgCuerrnt = reShape(imageModel, imgCuerrnt)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        imgCuerrnt = cv2.dilate(imgCuerrnt, kernel)
        imgCuerrnt = cv2.erode(imgCuerrnt, kernel)
        contours, hierarchy = cv2.findContours(imgCuerrnt, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours.sort(key=takeSecond, reverse=False)
        contours.sort(key=takeY, reverse=False)
        img_result = img_color[y1:y1 + h1, x1:x1 + w1]
        y1 = max(y1 - 20, 0)
        x1 = max(x1 - 20, 0)
        y2 = min(y1 + h1 + 40, img_color.shape[0] - 1)
        x2 = min(x1 + w1 + 40, img_color.shape[1] - 1)
        img_to_correct = img_color[y1:y2, x1:x2]
        # 以下将hough变换改成仿射变换，如果没有找到四个角点，就使用hough
        # showImg('img_to_correct1',img_to_correct)
        img_result, solution = correctImageByPerspectiveUseCornerOrHough(img_to_correct, img_result, flag_of_path)
        if solution == 'hough':
            continue
        flag_of_path = flag_of_path + 1
        listsOfImg.append([reShape(imageModel, img_result), solution])
        # listsOfCnts.append(contours)

    return listsOfImg


# get Avaiable ROI by solution bounding rectangular
def getAvaiableImgUseCnt(img_color, listsOfCntNormalOrder, imageModel, flag_of_path):
    listsOfImg = []
    shape = imageModel.shape
    for cnt in listsOfCntNormalOrder:
        (x, y, w, h) = cv2.boundingRect(cnt)
        img_test = img_color[y:y + h, w:w + h]
        # showImg('img_test',img_test)
        _, img_roi = correctImageByPerspectiveUseCnt(img_color, cnt, shape[1], shape[0])
        if img_roi is not None:
            flag_of_path = flag_of_path + 1
            listsOfImg.append([img_roi, 'cnt'])
    # print('listsOfImg',len(listsOfImg))
    return listsOfImg


# util func to load trained models,resue the train_step function of test_deblur
# esc_encoder_wokl_2_path : the path of esc encoder,this model to encode the sharp iamge context
# ebc_encoder_wokl_2_path : the path of ebc encoder,this model to encode the blur image contex
# eb_encoder_wokl_2_path : the path of eb encoder,this model to encode the blur
# gen_s_wokl_2_path :the path gen sharp image model ,this model to generator clear image by blur and image context
def setUtils(esc_encoder_wokl_2_path, ebc_encoder_wokl_2_path, eb_encoder_wokl_2_path,
             gen_s_wokl_2_path, tenClass_path):
    deblur_tool = DeBlur(1, esc_encoder_wokl_2_path, ebc_encoder_wokl_2_path, eb_encoder_wokl_2_path, gen_s_wokl_2_path)
    deblur_tool.set_buffer_size(1)
    class_tool = Classification(tenClass_path)
    class_tool.set_batch_size(15)
    return deblur_tool, class_tool


# result:the result of ROI identification，when one image get 15 numbers
# when the numbers of -1 more than 5,we diacard this image
def invalidate_index(result):
    result_invalidate_index = []
    shape = np.shape(result)
    for i in range(shape[0]):
        if result[i].count(-1) > 5:
            result_invalidate_index.append(i)
    return result_invalidate_index


# get the time on one ROI by 15 numbers and there confidences,five nums
# sub_result_list : the 15 numbers of one ROI
# sub_list_confidence :  thr confidence of every number
def getTheTime(sub_result_list, sub_list_confidence):
    result_one_time = []
    for i in range(5):
        nums_list = []
        nums_list.append(int(sub_result_list[i]))
        nums_list.append(int(sub_result_list[i + 5]))
        nums_list.append(int(sub_result_list[i + 10]))
        if nums_list.count(-1) >= 2:
            # print('first pass')
            pass
        elif nums_list.count(-1) == 1:
            if nums_list[0] == -1 and (nums_list[1] + 1) % 10 == nums_list[2]:
                if nums_list[1] == 0:
                    result_one_time.append(9)
                else:
                    result_one_time.append(nums_list[1] - 1)
            elif nums_list[1] == -1 and (nums_list[0] + 2) % 10 == nums_list[2]:
                result_one_time.append(nums_list[0])
            elif (nums_list[0] + 1) % 10 == nums_list[1]:
                result_one_time.append(nums_list[0])
        else:
            if (nums_list[0] + 1) % 10 == nums_list[1] and (nums_list[1] + 1) % 10 == nums_list[2] or \
                    (nums_list[0] + 2) % 10 == nums_list[2] or (nums_list[0] + 1) % 10 == nums_list[1]:
                result_one_time.append(nums_list[0])
            elif (nums_list[1] + 1) % 10 == nums_list[2]:
                if nums_list[1] == 0:
                    result_one_time.append(9)
                else:
                    result_one_time.append(nums_list[1] - 1)

        # if len(result_one_time) != i + 1:
        if len(result_one_time) == i:
            confidence_result = getTimeByConfidence(sub_list_confidence, i)
            # print('i is {},confidence_result is {}'.format(i,confidence_result))
            # print('sub_list_confidence[i] is {}'.format(sub_list_confidence))
            if confidence_result == -1:
                return result_one_time
            else:
                result_one_time.append(confidence_result)
    return result_one_time


# get time by confidence
def getTimeByConfidence(sub_list_confidence, i):
    sub_list_confidence = np.delete(sub_list_confidence, 0, axis=1)
    result_of_confidence = -1
    max_counter_of_index = 0
    max_confidence = 0.0
    counter = 0
    while counter <= 9:
        # print('counter is{}'.format(counter))
        if sub_list_confidence[i][counter] > 0.0:
            # print('a')
            sub_max_confidence = sub_list_confidence[i][counter]
            if sub_list_confidence[i + 5][(counter + 1) % 10] > 0.0:
                # print('b')
                sub_max_confidence = sub_max_confidence + sub_list_confidence[i + 5][(counter + 1) % 10]
                if sub_max_confidence > max_confidence:
                    # print('c')
                    max_confidence = sub_max_confidence
                    result_of_confidence = counter
                if sub_list_confidence[i + 10][(counter + 2) % 10] > 0.0:
                    # print('d')
                    sub_max_confidence = sub_max_confidence + sub_list_confidence[i + 10][(counter + 2) % 10]
                    if sub_max_confidence > max_confidence:
                        # print('e')
                        max_confidence = sub_max_confidence
                        result_of_confidence = counter
            elif sub_list_confidence[i + 10][(counter + 2) % 10] > 0.0:
                # print('f')
                sub_max_confidence = sub_max_confidence + sub_list_confidence[i + 10][(counter + 2) % 10]
                if sub_max_confidence > max_confidence:
                    # print('g')
                    max_confidence = sub_max_confidence
                    result_of_confidence = counter
        else:
            # print('h')
            sub_max_confidence = 0
            if sub_list_confidence[i + 5][(counter + 1) % 10] > 0.0 and sub_list_confidence[i + 10][
                (counter + 2) % 10] > 0.0:
                # print('i')
                sub_max_confidence = sub_list_confidence[i + 5][(counter + 1) % 10] + \
                                     sub_list_confidence[i + 10][(counter + 2) % 10]
                if sub_max_confidence > max_confidence:
                    # print('j')
                    max_confidence = sub_max_confidence
                    result_of_confidence = counter
        counter = counter + 1
    return result_of_confidence


def ocr_timestamp(img_file_path):
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200

    WIDTH_THRESH = 250
    HEIGHT_THRESH = 250

    ocr_time = ocr_result = None

    base_name = util.split_basename(img_file_path)
    img = cv2.imread(img_file_path)
    img_height, img_width, _ = img.shape
    if img_width < img_height:
        center = (img_width / 2, img_width / 2)
        M = cv2.getRotationMatrix2D(center, 90, 1.0)
        img = cv2.warpAffine(img, M, (img_height, img_width))
        cv2.imwrite(img_file_path, img)
        img_height, img_width, _ = img.shape

    MAX_WIDTH = 2000
    if img_width > MAX_WIDTH:
        img = cv2.resize(img, (MAX_WIDTH, int(img_height * MAX_WIDTH / img_width)))

    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_gray = cv2.threshold(gray, 100, 255, 0)
    edges = cv2.Canny(bin_gray, CANNY_THRESH_1, CANNY_THRESH_2)

    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[2], reverse=True)

    # cv2.imshow('ret', edges)
    cv2.waitKey(2000)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < WIDTH_THRESH:
            break
        if h < HEIGHT_THRESH:
            continue
        img_clipped = gray[y:y + h, x:x + w]
        ocr_result = tesseract.image_to_string(img_clipped, config='digit') + '\n'
        img_clipped = bin_gray[y:y + h, x:x + w]
        ocr_result += tesseract.image_to_string(img_clipped, config='digit') + '\n'
        img_clipped = img[y:y + h, x:x + w]
        ocr_result += tesseract.image_to_string(img_clipped, config='digit') + '\n'

    return ocr_time, ocr_result


# get time result or delay，use getAvaiableImgUseCnt or getAvaiableImgUseCorner to get ROI,
# then get the indicator kind declare
# img_color : get the time of this image record
# imageModel : model
# flag_of_path : the counter of image
# kind : two cases time result or delay
def pregetTimeresultOrDelay(img_color, imageModel, flag_of_path=0, kind='timeresult',listsOfImg_position=[], detectrou_tool=None):
    listsOfImg = []
    img_result = []
    img_result_position = listsOfImg_position
    listsOfImg_position_push_flag = False
    if len(listsOfImg_position) == 0:
        # print('listsOfImg_position is 0')
        listsOfImg_position_push_flag = True
        detectrou_tool.set_path_or_image(img_color, 'model_8.jpg')
        img_result,img_result_position = detectrou_tool.getTwoOrOne(img_color)
    else :
        # print('listsOfImg_position is 1')
        for x1,x2,y1,y2 in listsOfImg_position:
            img_result.append(img_color[y1:y2,x1:x2])
    # listsOfImg_position = []
    for img_sub,position in zip(img_result,img_result_position):
        shape_sub = img_sub.shape
        if shape_sub[1] / shape_sub[0] < 0.65 or shape_sub[1] / shape_sub[0] > 1.3:
            continue
        # showImg('listsOfImg_sub[0]', img_sub)
        listsOfCnt, listsOfCntNormalOrder = findROI(img_sub, imageModel, 100, 100, 30000, 140000, 1)
        # list_poionts = getAvaiableImgUseCorner(img_sub, listsOfCnt, 54, 36, 5, imageModel, flag_of_path)  # path,
        listsOfImg_sub =  getAvaiableImgUseCnt(img_sub, listsOfCntNormalOrder, imageModel, flag_of_path)  # path,
        if len(listsOfImg_sub) < 1:
            listsOfImg_sub = getAvaiableImgUseCorner(img_sub, listsOfCnt, 54, 36, 5, imageModel, flag_of_path)  # path, 
        if len(listsOfImg_sub) >= 1:
            listsOfImg.append(listsOfImg_sub[0])
            if True == listsOfImg_position_push_flag :
                listsOfImg_position.append(position)
            # img = listsOfImg_sub[0][0]#[10:listsOfImg_sub[0][0].shape[0]-10,10:listsOfImg_sub[0][0].shape[1]-10]
            # showImg('listsOfImg_sub[0]',img)

    return listsOfImg, listsOfImg_position


# use deblur model and class model to deal image
# listsOfImg :  this list include all ROI of one picture
# deblur_tool :  the deblur model
# class_tool :  the classification model to recongize the number
# contours_model_result :  roughly position of every number in ROI,by model image
# name_flag : counter of image
def getTimeresult(listsOfImg, deblur_tool, class_tool, contours_model_result, name_flag):
    to_save_dir = Path(python_path + '/Archive_to_split_2')
    if not to_save_dir.exists():
        os.makedirs(to_save_dir)
    to_save_sub_dir = Path(python_path + '/Archive_to_split_1')
    if not to_save_sub_dir.exists():
        os.makedirs(to_save_sub_dir)
    th = 0
    seconfOfPath = path.split('/')[-1]
    dst_path = []
    # img_color = cv2.imread(path)
    # listsOfCnt = findROI(img_color, imageModel, 300, 300, 50000, 140000, 1)
    # listsOfImg, listsOfCnts = getAvaiableImg(path, listsOfCnt, 54, 36, 5, imageModel)

    result_list_fiften_result = []
    list_confidence_fiften_result = []
    for imgzip in listsOfImg:
        img_ori = imgzip[0]
        result_list_fiften = []
        list_confidence_fiften = []
        a = 'a'
        if len(img_ori.shape) == 3:
            img_binary = thresholdImage(img_ori)
            # img_gery = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # rct, img_binary = cv2.threshold(img_gery, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # showImg('ori', img)
        name_flag = name_flag + 1
        for cnt in contours_model_result:
            x1, y1, w1, h1 = cv2.boundingRect(cnt)

            result_final = slideWindow(img_binary, x1 - 10, y1 - 10, x1 + 10, y1 + 10, w1 + 5, h1 + 5, 1)

            img_curent_sub1 = img_binary[y1:y1 + h1, x1:x1 + w1]
            if len(result_final) > 0:
                x1 = result_final[0][0]
                y1 = result_final[0][1]
                w1 = result_final[0][2]
                h1 = result_final[0][3]

            aear = w1 * h1

            img_curent_sub = img_ori[y1:y1 + h1, x1:x1 + w1]
            shape_sub = img_curent_sub.shape
            if not (shape_sub[0] < 30 or shape_sub[1] < 20 or shape_sub[0] >= 0.9 * img_binary.shape[0] or
                    shape_sub[1] >= 0.9 * img_binary.shape[1]):
                # print('junejune')
                # print(os.path.abspath('.'))
                pathToSave = str(to_save_sub_dir) + '/' + str(th) + '_' + seconfOfPath + '.jpg'
                # cv2.imencode('.jpg', img_curent_sub)[1].tofile(pathToSave)
                deblur_tool.set_path_or_image_to_deblur(None, img_curent_sub)
                deblur_tool.set_dateset()
                deblured_image = deblur_tool.to_get_result()

                pathToSave = str(to_save_sub_dir) + '/' + a + '_' + str(th) + '_' + seconfOfPath + '.jpg'
                # print(pathToSave)
                img_gery = np.array(deblured_image[0] * 0.5 + 0.5) * 255
                # cv2.imencode('.jpg', img_gery)[1].tofile(pathToSave)

                class_tool.set_path_or_image(None, img_gery)

                result_list, list_confidence = class_tool.print_out()
                result_list_fiften.append(result_list[0])
                list_confidence_fiften.append(list_confidence[0])
                # os.remove(str(to_save_sub_dir)+'/'+str(th)+'_'+seconfOfPath)
                # os.remove(str(to_save_sub_dir)+'/'+a+'_'+str(th)+'_'+seconfOfPath)
                th = th + 1
        result_list_fiften_result.append(result_list_fiften)
        list_confidence_fiften_result.append(list_confidence_fiften)
        # print('***********************')
        # print(result_list_fiften)
    # print(listsOfCnt)
    # file_to_remove = glob.glob(str(to_save_dir)+'/*.jpg')
    # os.remove(file_to_remove)
    # file_to_remove = glob.glob(str(to_save_sub_dir)+'/*.jpg')
    # os.remove(file_to_remove)
    # shutil.rmtree(to_save_sub_dir, True)
    # shutil.rmtree(to_save_dir, True)
    return result_list_fiften_result, list_confidence_fiften_result, name_flag, dst_path


# get the ROI image to cal delay
# listsOfImg : the list of all ROI in a picture,to find the ROI image in it
# cnt :  every position of number in on ROI
# thIfDelay :  thr counter of image
def getDelayResultImage(listsOfImg, cnt, thIfDelay=1):
    i = len(listsOfImg) - 1
    while i >= 0:
        if listsOfImg[i][1] == 'perspective' or listsOfImg[i][1] == 'cnt':
            img_binary = listsOfImg[i][0]
            if len(img_binary.shape) == 3:
                img_gery = cv2.cvtColor(img_binary, cv2.COLOR_RGB2GRAY)
                rct, img_binary = cv2.threshold(img_gery, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            x1, y1, w1, h1 = cv2.boundingRect(cnt)
            result_final = slideWindow(img_binary, x1 - 10, y1 - 10, x1 + 10, y1 + 10, w1 + 5, h1 + 5, 1)
            if len(result_final) > 0:
                x1 = result_final[0][0]
                y1 = result_final[0][1]
                w1 = result_final[0][2]
                h1 = result_final[0][3]
            img_curent_sub = img_binary[y1:y1 + h1, x1:x1 + w1]
            counter_white = 0
            for x in range(img_curent_sub.shape[0]):
                for y in range(img_curent_sub.shape[1]):
                    if img_curent_sub[x][y] > 0:
                        counter_white = counter_white + 1
            if counter_white / (img_curent_sub.shape[0] * img_curent_sub.shape[1]) < 1:
                # showImg('img_roi'+str(thIfDelay),listsOfImg[i][0])#
                # print('junejune')
                return listsOfImg[i][0].copy(), thIfDelay + 1
        i = i - 1
    return None, thIfDelay + 1


# get the time of this image present,call the function  invalidate_index and getTheTime and so on
# return two timestampes of original and receiver
def getResultRromList(result_list, list_confidence):
    shape = np.shape(result_list)
    flagOfTrueOrFalse = 0
    result_of_time = []
    result_invalidate_index = 0
    if shape[0] == 1:
        # #print('first')
        return 100000, result_of_time, flagOfTrueOrFalse
    if shape[0] >= 2:
        result_invalidate_index = invalidate_index(result_list)
        # print('result_invalidate_index is {}'.format(result_invalidate_index))
    if shape[0] - len(result_invalidate_index) < 2:
        # print('second')
        return 100000, result_of_time, flagOfTrueOrFalse
    index_of_result_invalidate_index = 0
    for i in range(shape[0]):
        if index_of_result_invalidate_index < len(result_invalidate_index) and \
                i == result_invalidate_index[index_of_result_invalidate_index]:
            index_of_result_invalidate_index = index_of_result_invalidate_index + 1
            continue
        else:
            # print('result_list[i] is {}'.format(result_list[i]))
            result_time = getTheTime(result_list[i], list_confidence[i])
            # print('result_time is {}'.format(result_time))
            # print('i:{},result_time: {}'.format( i,result_time))
            if len(result_time) == 5 and result_time not in result_of_time:
                result_of_time.append(result_time)
    if len(result_of_time) < 2:
        # print('third')
        return 100000, result_of_time, flagOfTrueOrFalse
    duliang = 10000
    first_num = 0
    second_num = 0
    for i in range(len(result_of_time)):
        flag = True
        for j in range(len(result_of_time[0]) - 1):
            if (result_of_time[i][j] + 1) % 10 != result_of_time[i][j + 1]:
                flag = False
                break
        if flag:
            flagOfTrueOrFalse = flagOfTrueOrFalse + 1
    for i in range(len(result_of_time[0])):
        first_num = first_num + result_of_time[0][i] * duliang
        second_num = second_num + result_of_time[1][i] * duliang
        duliang = duliang / 10
    result_final = first_num-second_num#(first_num + 100000 - second_num) % 100000
    print('first_num : {},second_num : {}, result : {}'.format(first_num, second_num, result_final))
    return result_final, result_of_time, flagOfTrueOrFalse


# to return the current image for the next time to cal delay,and delay times
# return value:
# currentImage :the current image for the next time to cal delay
# pre_time : the timestampe of current image
# dealy_ms : the delay betwen now and the previous frame
# counter_all ：  counter of image
from PIL import Image
def hanming(A):
    resize_width = 9
    resize_height = 8
    # A = Image.open(path)
    grayscale_A = A.convert("L")
    pixels = list(grayscale_A.getdata())
    difference = []
    for row in range(resize_height):    
        row_start_index = row * resize_width    
        for col in range(resize_width - 1):        
                left_pixel_index = row_start_index + col
                difference.append(pixels[left_pixel_index] > pixels[left_pixel_index + 1])
    return difference
def getRidofBlackBound1(img1):
    shape = img1.shape
    begin_x = 0
    end_x = 0
    begin_y = 0
    end_y = 0
    for i in range(img1.shape[0]):
        if np.bincount(np.array(img1[i]))[-1] == img1.shape[1]:
            begin_x = i
				#print('begin_x',begin_x)
            break
    for i in range(img1.shape[0]):
        if np.bincount(np.array(img1[img1.shape[0]-i-1]))[-1] == img1.shape[1]:
            end_x = shape[0]-i
            break
    img1 = img1[begin_x:end_x,0:img1.shape[1]]

    for i in range(img1.shape[1]):
        if np.bincount(np.array(img1[:,i]))[-1] == img1.shape[0]:
            begin_y = i
            break
    for i in range(img1.shape[1]):
        if np.bincount(np.array(img1[:,img1.shape[1]-i-1]))[-1] == img1.shape[0]:
            end_y = shape[1]-i
            break
    img1 = img1[0:shape[0],begin_y:end_y]
    size = (shape[1], shape[0])
    img1 = cv2.resize(img1, size, interpolation=cv2.INTER_LINEAR)
    return img1

# #四分之一图像找到
# def findRedBeginx(img,start_x,start_y):
#     shape = img.shape
#     begin_y = [-1,-1]
#     end_y = [-1,-1]
#     begin_x = [-1,-1]
#     end_x = [-1,-1]

#     for i in range(int(shape[0]/4)):
#         for j in range(int(shape[1]/4)):
#             if img[i+start_x][j+start_y] == [255,0,0]:
#                 begin_x  = [i,j]
#                 break
#         if begin_x[0]  != -1:
#             break
#     for j in range(int(shape[1]/4)):
#         for i in range(int(shape[0]/4)):





def getRidofBlackBound(img1):
    shape = img1.shape
    begin_x = 0
    end_x = shape[0]
    begin_y = 0
    end_y = shape[1]
    i = 10
    # print(img1)
    while i >= 0:
    # for i in range(img1.shape[0]):
        # print('np.bincount(np.array(img1[i,20:-20]))[-1]',img1.shape[1],np.bincount(np.array(img1[i,20:-20])))
        if np.bincount(np.array(img1[i,20:-20]))[1] != img1.shape[1]-40:
            begin_x = i
            # print('begin_x',begin_x)
            break
        # print('now')
        i = i-1
    # for i in range(img1.shape[0]):
    i = 10

    while i >= 0:
        if np.bincount(np.array(img1[img1.shape[0]-i-1,20:-20]))[1] != img1.shape[1]-40:
            end_x = shape[0]-i
            # print('end_x',end_x)
            break
        i = i-1
    # img1 = img1[begin_x:end_x,0:img1.shape[1]]

    i = 10
    while i >= 0:
        if np.bincount(np.array(img1[20:-20,i]))[1] != img1.shape[0]-40:
            begin_y = i
            # print('begin_y',begin_y)
            break
        i = i-1

    i = 10
    while i >= 0:
        if np.bincount(np.array(img1[20:-20,img1.shape[1]-i-1]))[1] != img1.shape[0]-40:
            end_y = shape[1]-i
            # print('end_y',end_y)
            break
        i = i-1

    # print('hehhe')
    img1 = img1[begin_x:end_x,begin_y:end_y]
    size = (shape[1], shape[0])
    img1 = cv2.resize(img1, size, interpolation=cv2.INTER_LINEAR)
    # print(img1)
    return img1
def pointByX(a):
    return a[0]
def pointByY(a):
    return a[1]
import imutils

# def toFindFourRedPoints(img):
#     shape = img.shape
#     img = img[15:img.shape[0]-15,15:img.shape[1]-15]
#     for i in range(shape[0]):
#         for j in range(shape[1]):
def getBlackCenterPoint(img,point1,point2):
    i = point1[0]
    j = point1[1]
    shape = img.shape
    point_to_return = [-1,-1]
    min_sum = 255*9
    while i < point2[0]:
        while j < point2[1]:
            # if i-1<0 or i+1>=shape[0] and( j-1<0 or j+1>=shape[1]):
            #     j = j+1
            #     i = i+1
            #     continue
            # if i-1<0 or i+1>=shape[0] :
            #     i = i+1
            #     continue
            # if j-1<0 or j+1>=shape[1]:
            #     j = j+1
            #     continue
            # print(img[i-1][j-1],img[i-1][j],img[i-1][j+1],img[i][j-1],img[i][j],img[i-1][j+1],img[i+1][j-1],img[i+1][j],img[i+1][j+1])
            sum = int(img[i-1][j-1])+int(img[i-1][j])+int(img[i-1][j+1])+int(img[i][j-1])+int(img[i][j])+int(img[i-1][j+1])+int(img[i+1][j-1])+int(img[i+1][j])+int(img[i+1][j+1])
            # print('sum',sum,i,j)
            if sum < min_sum:
                min_sum = sum
                point_to_return[0] = i
                point_to_return[1] = j
            j=j+1
        i = i+1
        j = point1[1]
    return point_to_return[0],point_to_return[1]

def getFourPoints(img,begin_x,begin_y,x_or_y,d=30):
    # print('begin_x,begin_y',begin_x,begin_y)
    list_points = []
    shape = img.shape
    x_1_1 = -1
    x_1_2 = -1
    if x_or_y == 0:
        i = begin_x
    else:
        i = begin_y
    list_now = []

    while i < shape[x_or_y]:
        if x_or_y == 0:
            list_now = img[i,begin_y:min(begin_y+d,shape[1])].tolist()
            list_now.sort()
        else :
            list_now = img[begin_x:min(begin_x+d,shape[0]),i].tolist()
            list_now.sort()
        # print('list_now',list_now)
        if list_now [0] < 230 and -1 == x_1_1:#4个值小于230
            x_1_1 = i
        elif list_now [0] >= 230 and -1 != x_1_1:
            x_1_2 = i
            break
        i = i+1
    # print('x_1_1',x_1_1)
    # print('x_1_2',x_1_2)
    return (int((x_1_1+x_1_2)/2))


    # return list_points
def toCrectImg(img,list_point= [],frame_counter=-1):
    img = img[5:img.shape[0]-5,5:img.shape[1]-5]#shape[1]
    img_copy = img.copy()
    img = thresholdImage(img)
    # img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # if frame_counter == 84:
    #     showImg('img--',img)
    shape = img.shape
    if len(list_point) == 0:
        list_point = []#,p_1_2 = getBlackCenterPoint(img,[1,1],[31,31])#
        p_1_1 = getFourPoints(img,0,0,0)
        p_1_2 = getFourPoints(img,0,0,1)
        list_point.append([p_1_2,p_1_1])
        # if frame_counter == 84:
        #     # print('p_1_4',p_1_1,p_1_2)
        #     print('p_1_1',p_1_1,p_1_2)
        img[p_1_1][p_1_2] = 255#255,0,0]

        #
        # p_1_1,p_1_2 = getBlackCenterPoint(img,[shape[0]-24,1],[shape[0]-1,30])#getFourPoints(img,0,0,0)
        
        p_1_1 = getFourPoints(img,shape[0]-24,0,0)
        p_1_2 = getFourPoints(img,shape[0]-24,0,1,d=20)
        list_point.append([p_1_2,p_1_1])
        # if frame_counter == 84:
        #     # print('p_1_4',p_1_1,p_1_2)
        #     print('p_1_2',p_1_1,p_1_2)
        img[p_1_1][p_1_2] = 255#0,0,255]
        # p_1_1,p_1_2 = getBlackCenterPoint(img,[shape[0]-24,shape[1]-30],[shape[0]-1,shape[1]-1])#getFourPoints(img,0,0,0)
        # print('shape',shape)
        p_1_1 = getFourPoints(img,shape[0]-23,shape[1]-26,0)
        p_1_2 = getFourPoints(img,shape[0]-23,shape[1]-26,1)#,d=20)
        list_point.append([p_1_2,p_1_1])
        # if frame_counter == 84:
        #     # print('p_1_4',p_1_1,p_1_2)
        #     print('p_1_3',p_1_1,p_1_2)
        img[p_1_1][p_1_2] = 255#255,255,255]

        # 右上
        p_1_1 = getFourPoints(img,0,shape[1]-40,0)
        p_1_2 = getFourPoints(img,0,shape[1]-40,1)
        # p_1_1,p_1_2 = getBlackCenterPoint(img,[1,shape[1]-31],[31,shape[1]-1])#getFourPoints(img,0,0,0)
        
        list_point.append([p_1_2,p_1_1])
        img[p_1_1][p_1_2] = 255#0,255,0]
        # showImg('dot',img,time=100000)
            
    quad_pts = np.array(list_point,
                        np.float32)
    # squre_pts = np.array(second_list,
    #                      np.float32)

    squre_pts = np.array([[0, 0],
                          [0, shape[0]],#shape[0]
                          [shape[1], shape[0]],
                          [shape[1], 0]],# shape[1]
                         np.float32)
    transform1 = cv2.getPerspectiveTransform(quad_pts,squre_pts)
    img = cv2.warpPerspective(img, transform1, (img.shape[1], img.shape[0]))
    image_color = cv2.warpPerspective(img_copy, transform1, (img.shape[1], img.shape[0]))
    
    img = img[20:img.shape[0]-20,0:img.shape[1]]
    image_color = image_color[20:image_color.shape[0]-20,0:image_color.shape[1]]
    rct, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # if frame_counter == 230:
    #     showImg('img230',img)
    #     showImg('image_color230',image_color)
    return img,list_point,image_color

def recog(img,model,class_tool=None,deblur_tool=None):
    img = img[0: img.shape[0],int(img.shape[1]/5*2):img.shape[1]]
    model = model[0: model.shape[0],int(model.shape[1]/5*2):model.shape[1]]
    # showImg('img_color',img)
    # showImg('model_color',model)
    true_of_flag=0
    false_of_flag=0
    shape = img.shape
    jiange_1 = int(shape[1]/3)
    jiange_2 = int(shape[0]/3)
    for i in range(3):
        for j in range(3):
            img_gery_1 = img[jiange_2*j:jiange_2*(j+1),jiange_1*i:jiange_1*(i+1)]

            img_gery_2 = model[jiange_2*j:jiange_2*(j+1),jiange_1*i:jiange_1*(i+1)]
            # showImg('img_gery_1',img_gery_1)
            # showImg('img_gery_2',img_gery_2)

            if (img_gery_1.ndim ==img_gery_2.ndim ):
                simm = compare_ssim(img_gery_1,img_gery_2,multichannel=True)
                # print('simm',simm)
                if simm >= 0.90:
                    true_of_flag =true_of_flag+1
                    continue
            result_1 = -2
            result_2 = -3
            if img_gery_1.ndim >= 3:
                deblur_tool.set_path_or_image_to_deblur(None, img_gery_1)
                deblur_tool.set_dateset()
                img_gery_1 = deblur_tool.to_get_result()
                img_gery_1 = np.array(img_gery_1[0] * 0.5 + 0.5) * 255
                # showImg('img_gery_1',img_gery_1)#,time=2000)
                class_tool.set_path_or_image(None, img_gery_1)
                result_list_1, list_confidence_1 = class_tool.print_out()
                result_1 = result_list_1[0]
            if img_gery_2.ndim >= 3:
                deblur_tool.set_path_or_image_to_deblur(None, img_gery_2)
                deblur_tool.set_dateset()
                img_gery_2 = deblur_tool.to_get_result()
                img_gery_2 = np.array(img_gery_2[0] * 0.5 + 0.5) * 255
                # showImg('img_gery_2',img_gery_2)#,time=2000)
                class_tool.set_path_or_image(None, img_gery_2)
                result_list_2, list_confidence_2 = class_tool.print_out()
                result_2 = result_list_2[0]
            # print('result_1',result_1)
            # print('result_2',result_2)

            if result_2 == result_1:#or simm > 0.90:
                true_of_flag = true_of_flag + 1
                # print('相同')
            else :
                false_of_flag = false_of_flag+1
                # print('不同')
                return true_of_flag,false_of_flag
    # print(true_of_flag , false_of_flag)
    return true_of_flag, false_of_flag

def Loss3(model,img,class_tool=None,deblur_tool=None,frame_counter = 0,strat_list = [],index_1 = 0):
    # if frame_counter < 1396:
    #     return False
    # if frame_counter == 0:
    #     return False
    t_2 = 9999999
    img_copy = img.copy()
    img,list_point_current,img_color  = toCrectImg(img,frame_counter=frame_counter)
    model,list_point_before, model_color = toCrectImg(model,frame_counter=frame_counter)
   
    img = img[0: img.shape[0],int(img.shape[1]/5*2):img.shape[1]]
    model = model[0: model.shape[0],int(model.shape[1]/5*2):model.shape[1]]

    t = np.sum(np.sqrt(model-img))
    flag = False
    if t < 10000:
        # print('t<10000')
        return True
    elif t < 30000:
        true_of_flag,false_of_flag = recog(img_color, model_color, class_tool=class_tool, deblur_tool=deblur_tool)
        # print('true_of_flag',true_of_flag)
        return true_of_flag == 9
    # print('t=',t)
    return False


def toRecordDealy(listsOfImg, cnt, thIfDelay, preImage, pre_time,pre_id, dealy_ms, counter_all,
                  frame_end_time, couter_frame_wode, class_tool,deblur_tool,strat_list = [],index_ = 0,the_second_flag=False ):
    print('couter_frame_wode is {},now_time = {}'.format(couter_frame_wode,frame_end_time))
    # couter_frame_wode = couter_frame_wode+1
    # print('now_cnt',cnt)
    currentImage, thIfDelay = getDelayResultImage(listsOfImg, cnt, thIfDelay)
    current_time = 0.0
    end_time = 0.0
    if currentImage is None:
        print('because currentImage is None')
        return None, pre_time,pre_id, dealy_ms, counter_all,1,the_second_flag
    if True == the_second_flag:
        # 下一帧不再分析
        the_second_flag = False
        return currentImage.copy(),pre_time, pre_id,dealy_ms, counter_all,1,the_second_flag
    
    # currentImage = cv2.resize(currentImage, (32, 32))
    # showImg('preImage',preImage)
    # image1_difference = hanming(Image.fromarray(currentImage))#cv2.cvtColor(currentImage,cv2.COLOR_BGR2RGB)))
    # image2_difference = hanming(Image.fromarray(preImage))#cv2.cvtColor(preImage,cv2.COLOR_BGR2RGB)))
    # hamming_distance = 0
    # for index, img1_pix in enumerate(image1_difference):
    #     img2_pix = image2_difference[index]
    #     if img1_pix != img2_pix:
    #         hamming_distance += 1
    # print('hanming',hamming_distance)
    # loss1 = L2Loss1(preImage, currentImage)
    # print('loss_1 = ',loss1)
    # loss2 = L2Loss2(preImage,currentImage,frame_end_time)
    # print('loss_2 = ',loss2)
    loss3 = Loss3(preImage,currentImage,class_tool,deblur_tool,couter_frame_wode)#,strat_list=strat_list,index_1=index_)
    # print('loss_3 = ',loss3)
    counter_all = counter_all + 1

    if pre_time != -1:
        # capture.get(cv2.CAP_PROP_POS_MSEC))
        # end_time = int(capture.get(cv2.CAP_PROP_POS_MSEC))
        current_time = frame_end_time
        # print('loss={},area={}'.format(loss,preImage.shape[0]*preImage.shape[1]))
        if  False == loss3 :#np.sum(loss2) > 1500:#(loss2[0]> 1000 or loss2[1]> 1000  or loss2[2]> 1000):#(loss1[0]<0.5 or loss1[1]<0.5 or loss1[2] <0.5 ) and#(np.sum(loss1)-np.sum(pre_chazhi))>100:# np.sum(loss)/3 >=50:#loss[0] > 20 and loss[1]>20 and loss[2]>20:# np.sum(loss)/((preImage.shape[0]-30)*(preImage.shape[1]-30)) > 0.05:#8000:
            # the_second_flag = True
            delay = current_time - pre_time
            # if delay >= 99:
            #
            #     # print('100 current_time is {},pre_time is {}'.format(current_time,pre_time))
            #     print('100 pre_id is {},current_id is {}'.format(pre_id,couter_frame_wode-1))
            #
            #     dealy_ms[3] = dealy_ms[3] + delay

            if delay >= 200:
                # print('200 pre_id is {},current_id is {}'.format(pre_id,couter_frame_wode-1))
                
                dealy_ms[0] = dealy_ms[0] + delay
            if delay >= 300:
                # print('300 pre_id is {},current_id is {}'.format(pre_id,couter_frame_wode-1))
                
                dealy_ms[1] = dealy_ms[1] + delay
            if delay >= 600:
                # print('600 current_time is {},pre_time is {}'.format(current_time,pre_time))
                # print('600 pre_id is {},current_id is {}'.format(pre_id,couter_frame_wode-1))
                
                dealy_ms[2] = dealy_ms[2] + delay
            # int(capture.get(cv2.CAP_PROP_POS_MSEC))
            pre_time = frame_end_time
            pre_id = couter_frame_wode
        else:
            pass
            # print('\n')#'current_time is {},pre_time is {}'.format(current_time,pre_time))
    else:
        first_delay_flag = False
        # capture.get(cv2.CAP_PROP_POS_MSEC))
        pre_time = frame_end_time
        begin_time = pre_time

    # print('counter_all is {}'.format(counter_all))
    # print('pre_time is {}, current_time is {}'.format(current_time))
    # print('this is the {} frame,loss with preframe is {}'.format(counter_all,loss))
    # preImage = currentImage.copy()
    # flag_of_path = flag_of_path+3

    return currentImage.copy(), pre_time, pre_id,dealy_ms, counter_all,1,the_second_flag

def binary_find_the_same_frame(first_reserve,current_frame,cnt,class_tool,deblur_tool,detectrou_tool,flag = 'headortail',listsOfImg_position=[],pre_time= -1,imageModel=None,must_tail=False):
    
    # now_image = first_reserve[0][1]
    # listsOfImg, listsOfImg_position = pregetTimeresultOrDelay(now_image, imageModel,listsOfImg_position=listsOfImg_position,detectrou_tool=detectrou_tool)
    # now_image, thIfDelay = getDelayResultImage(listsOfImg, cnt)#contours_model_result[0])
        
    # loss3 = Loss3(current_frame,now_image,class_tool,deblur_tool)#,couter_frame_wode)#,strat_list=strat_list,index_1=index_)
    # if loss3 == True:
    #     return first_reserve[-1:],first_reserve[0][0]

    begin = 0
    end = len(first_reserve)-1
    # print('begin_time,end_time,flag',first_reserve[0][0],first_reserve[-1][0],flag)
    # print('must_tail',must_tail)
    while begin+1 < end:
        # print('binary_find_the_same_frame')
        middle  = int((begin+end)/2)
        # if flag == 'tail':
        #         if must_tail == False and abs(pre_time-first_reserve[middle][0]) < 100:
        #             return first_reserve,-1
        now_image = first_reserve[middle][1]
        listsOfImg, listsOfImg_position = pregetTimeresultOrDelay(now_image, imageModel,listsOfImg_position=listsOfImg_position,detectrou_tool=detectrou_tool)
        now_image, thIfDelay = getDelayResultImage(listsOfImg, cnt)#contours_model_result[0])
        loss3 = Loss3(current_frame,now_image,class_tool,deblur_tool)#,couter_frame_wode)#,strat_list=strat_list,index_1=index_)

        if loss3 == True:
            if flag == 'tail':
                begin = middle
            else:
                end = middle
        else:
            # if flag == 'tail':

            if flag == 'tail' :
                if  must_tail == False and abs(pre_time-first_reserve[middle][0]) < 200:
                    return first_reserve,-1
                end = middle
            else :
                begin = middle
    # 返回值是将要处理的数组,和当前的的时间，会更新成pre_time
    # if True == must_tail :
    #     first_reserve_list.append(first_reserve[0:end-1])
    # print('end=',end)
    return first_reserve[end:],first_reserve[end][0]

import csv
if __name__ == '__main__':
    # 拿到表格的数据想从而进行筛选数据

    # f = open('/Users/xhj/Downloads/result/5-result.csv')
    # reader = csv.reader(f)
    # array_now = np.array(list(reader))
    # print(array_now)
    # strat = array_now[:,1]
    # end = array_now[:,2]
    # strat_list =[]
    # end_list =[]
    # strat = list(strat)[1,:]
    # end = list(end)[1,:]
    # print(strat)
    # print(end)
    # for i in range(len(strat)):
    #     if i == 0 or i == len(strat)-1:
    #         continue
    #     a = int(float(strat[i].strip()))
    #     b = int(float(end[i].strip()))
    #     # print(int(float(strat[i])),  int(float(end[i])))
    #     strat_list.append([a,b])#([ int(float(strat[i])), int(float(end[i]))])
    # # print('strat_list',strat_list)
    python_path = os.environ["PYTHONPATH"].split(':')[0]
    path = '/Users/xhj/Downloads/Communication-360_640-800-15-UL50_FREEZE_RATE.mp4'
    
    result_half = []
    head_and_tail = []
    first_reserve_list = []

    def update_pre_and_first_reserve(first_reserve,pre_image,currentImage,pre_time,class_tool,deblur_tool,detectrou_tool,cnt,dealy_ms,listsOfImg_position=[],imageModel=None,first_200=False):#,last_200 =False):
        # showImg('pre_image',pre_image)
        # showImg('currentImage',currentImage)
        loss3 = Loss3(pre_image,currentImage,class_tool=class_tool,deblur_tool=deblur_tool)#,couter_frame_wode)#,strat_list=strat_list,index_1=index_)
        
        # print('update_pre_and_first_reserve loss3june',loss3,first_reserve[0][0],first_reserve[-1][0])
        # if first_reserve[0][0] == 90260 and first_reserve[-1][0] == 90426:
        #     showImg('pre_image',pre_image)
        #     showImg('currentImage',currentImage)
        # True表示是相同的帧，false表示是不同的帧内容
        if False == loss3:
            # 求第一帧的尾
            first_reserve,time_now = binary_find_the_same_frame(first_reserve, pre_image,cnt,class_tool,deblur_tool,detectrou_tool,flag='tail',listsOfImg_position=listsOfImg_position,pre_time=pre_time,imageModel=imageModel,must_tail=first_200)
            if first_200 == True:
                # print('pre_time,time_now',pre_time,time_now)
                first_reserve_list.append((pre_time,time_now,pre_image))
            
            # print('len(first_reserve)',len(first_reserve))
            delay = time_now - pre_time

            if False == first_200:
                if delay >= 200:
                    dealy_ms[0] = dealy_ms[0] + delay
                    print('200 pre_time is {},time_now is {}'.format(pre_time,time_now))

                if delay >= 300:
                    dealy_ms[1] = dealy_ms[1] + delay
                    print('300 pre_time is {},time_now is {}'.format(pre_time,time_now))

                if delay >= 600:
                    print('600 pre_time is {},time_now is {}'.format(pre_time,time_now))
                    dealy_ms[2] = dealy_ms[2] + delay
            first_200 = False
            # first_reserve最后一帧的roi， 也就是将要求头的那一阵
            pre_image = currentImage.copy()
            # 更新数组，找到和pre_image相同的帧，将其之前的帧都删除掉
            # print('second')
            # 求最后一帧的头
            # 如果现在首尾相同的话直接返回
            now_image = first_reserve[0][1]
            listsOfImg, listsOfImg_position = pregetTimeresultOrDelay(now_image, imageModel,listsOfImg_position=listsOfImg_position,detectrou_tool=detectrou_tool)
            now_image, thIfDelay = getDelayResultImage(listsOfImg, cnt)#contours_model_result[0])
                
            loss3 = Loss3(currentImage,now_image,class_tool,deblur_tool)#,couter_frame_wode)#,strat_list=strat_list,index_1=index_)
            
            if loss3 == True:
                pre_time = first_reserve[0][0]
                first_reserve = first_reserve[-1:]
                
                # print('first_reserve',len(first_reserve))
                return first_reserve,pre_image,pre_time,dealy_ms,first_200
                # return first_reserve[-1:],first_reserve[0][0]
            # 求最后一帧的头
            first_reserve,time_now = binary_find_the_same_frame(first_reserve,currentImage,cnt,class_tool,deblur_tool,detectrou_tool,flag='head',listsOfImg_position=listsOfImg_position,imageModel=imageModel)#,must_tail=(first_200 or last_200))
            pre_time = time_now#first_reserve[0][0]
        else :# 如果是相同的,将当前数组全部删除，记录下当前的时间
            # find_head_of_first_reserve = True
            # print('相同')
            first_reserve = first_reserve[-1:]
            # pre_image = currentImage.copy()
            # print(first_reserve[0][0])
        # print('pre_time',pre_time)
        return first_reserve,pre_image,pre_time,dealy_ms,first_200

    def tuple_first(elem):
        return elem[0]

    detectrou_tool = testbk.DetectROI(net_path=python_path + '/two_rec_2.h5')
        
    deblur_tool, class_tool = setUtils(
            python_path + '/esc_encoder_wokl_3.h5',
            python_path + '/ebc_encoder_wokl_3.h5',
            python_path + '/eb_encoder_wokl_3.h5',
            python_path + '/gen_s_wokl_3.h5',
            python_path + '/tenClass_cpu.h5')
    def half_of_video_deal(begin_frame_counter,end_frame_counter,path):
        list_point = []
        import datetime
        
        imageModel, contours_model_result = getModelJEPGAnd15Positions("model_8.jpg")
        # '/Users/xhj/Downloads/VID_20200814_211526.mp4')#'/Users/xhj/Downloads/VID_20200814_211526.mp4')
        capture = cv2.VideoCapture(path)
        name_flag = 0
        flag_of_path = 0
        preImage = imageModel.copy()
        path_of_all = glob.glob('/Users/xhj/Downloads/192423/*.jpg')
        counter_all = 0
        first_delay_flag = True
        pre_time = -1.0
        pre_id = -1
        current_time = 0.0
        dealy_ms = [0.0, 0.0, 0.0]

        begin_time = 0.0
        end_time = 0.0
        thIfDelay = 1
        counter_frame = 0
        listsOfImg_position = []
        couter_frame_wode = 0
        pre_chazhi = []
        index = 0
        max_time = -1
        counter_9 = 0
        counter_8 = 0
        the_second_flag = False
        istsOfImg_record = []
        listsOfImg = []
        print(datetime.datetime.now())
        first_reserve = []
        # second_reserve = []
        pre_image = None
        pre_time = -1
        # find_head_of_first_reserve = False
        counter_of_frame = -1

        first_200 = True
        # last_200 = False

        while capture.isOpened():

            if len(first_reserve) == 0 or end_time - first_reserve[0][0] < 200:
                frame_exists, current_frame = capture.read()
                counter_of_frame = counter_of_frame+1
                if counter_of_frame < begin_frame_counter :
                    continue

                if counter_of_frame > end_frame_counter:
                    break

                if not frame_exists:
                    print('because end')
                    break

                # 记录下之前的帧
                end_time = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                first_reserve.append((end_time,current_frame))
                if pre_time == -1:
                    pre_time = end_time
                
                if pre_image is None:
                    listsOfImg, listsOfImg_position = pregetTimeresultOrDelay(current_frame, imageModel, flag_of_path,listsOfImg_position,detectrou_tool=detectrou_tool)
                    pre_image,thIfDelay = getDelayResultImage(listsOfImg, contours_model_result[0])

            else :
                # to get the fitst image of first_reserve
                # 如果已经满了200ms的帧,计算出最后这一帧的roi：currentImage，然后计算此帧的尾和头在update_pre_and_first_reserve中
                listsOfImg, listsOfImg_position = pregetTimeresultOrDelay(current_frame, imageModel, flag_of_path,listsOfImg_position,detectrou_tool=detectrou_tool)
                currentImage, thIfDelay = getDelayResultImage(listsOfImg, contours_model_result[0])
                # 接下来计算pre_image 和currentImage 之间的相似度差值
                if pre_image is None :
                    print('because pre_image is None')
                    break
                if  currentImage is None  :
                    print('because currentImage is None')
                    break
                # 计算200毫秒的数组中首尾是否相同，
                #pre_image 是数组的第一帧的roi，currentImage 是数组最后一帧的roi
                first_reserve,pre_image,pre_time,dealy_ms,first_200 = update_pre_and_first_reserve(first_reserve,pre_image,currentImage,pre_time,class_tool,deblur_tool,detectrou_tool,contours_model_result[0],dealy_ms,listsOfImg_position=listsOfImg_position,imageModel=imageModel,first_200=first_200)
                
            
            
            # print('end_time',end_time)
           
                # first_200 = False
                
        print(datetime.datetime.now())
        print(dealy_ms)
        print(end_time)
        print(path)
        print(dealy_ms[0]/end_time)
        print(dealy_ms[1]/end_time)
        print(dealy_ms[2]/end_time)
        print('final', counter_8, counter_9)
        result_half.append(dealy_ms)
        # listsOfImg, listsOfImg_position = pregetTimeresultOrDelay(first_reserve[0][1], imageModel, flag_of_path,listsOfImg_position,detectrou_tool=detectrou_tool)
        # pre_image, thIfDelay = getDelayResultImage(listsOfImg, contours_model_result[0])
        # 最后一次比较得到pre_image 和 pre_time
        listsOfImg, listsOfImg_position = pregetTimeresultOrDelay(first_reserve[-1][1], imageModel, flag_of_path,listsOfImg_position,detectrou_tool=detectrou_tool)
        currentImage, thIfDelay = getDelayResultImage(listsOfImg, contours_model_result[0])
        first_reserve,pre_image,pre_time,dealy_ms,_ = update_pre_and_first_reserve(first_reserve,pre_image,currentImage,pre_time,class_tool,deblur_tool,detectrou_tool,contours_model_result[0],dealy_ms,listsOfImg_position=listsOfImg_position,imageModel=imageModel)#,last_200=True)

        # 最尾部的
        first_reserve_list.append((pre_time,first_reserve[-1][0],currentImage))

        return


    cap = cv2.VideoCapture(path)
    frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frame_counter',frame_counter)
    cap.release()

    th1 = threading.Thread(target= half_of_video_deal, args=(0,int(frame_counter/2),path))
    th1.start()
    th2 = threading.Thread(target= half_of_video_deal, args=(int(frame_counter/2)+1,int(frame_counter),path))
    th2.start()
    th1.join()
    th2.join()
    # half_of_video_deal(int(frame_counter/2)+3,int(frame_counter),path)

    dealy_ms=[0.0,0.0,0.0]
    first_reserve_list.sort(key=tuple_first)
    print('first_reserve_list',first_reserve_list)
    i = 0
    while i < len(first_reserve_list)-2:
        if i == 0:
            delay = first_reserve_list[i][1]-first_reserve_list[i][0]
        else :
            pre_image = first_reserve_list[i][2]
            currentImage = first_reserve_list[i+1][2]
            # showImg('pre_image',pre_image)
            # showImg('currentImage',currentImage)
            loss3 = Loss3(pre_image,currentImage,class_tool,deblur_tool)#,couter_frame_wode)#,strat_list=strat_list,index_1=index_)
            print('now time',first_reserve_list[i+1][1],first_reserve_list[i][0])
            if loss3 == True:
                delay = first_reserve_list[i+1][1]-first_reserve_list[i][0]
            else :
                delay = max(first_reserve_list[i+1][1]-first_reserve_list[i+1][0], first_reserve_list[i][1]-first_reserve_list[i][0])
        if delay >= 200:
            dealy_ms[0] = dealy_ms[0] + delay
            # print('200 pre_time is {},time_now is {}'.format(pre_time,time_now))

        if delay >= 300:
            dealy_ms[1] = dealy_ms[1] + delay
            # print('300 pre_time is {},time_now is {}'.format(pre_time,time_now))

        if delay >= 600:
            # print('600 pre_time is {},time_now is {}'.format(pre_time,time_now))
            dealy_ms[2] = dealy_ms[2] + delay
        i = i+1
    result_half.append(dealy_ms)
    print(result_half)
    print('now two thread end!')
    

    # loss3 = Loss3(pre_image,currentImage,class_tool,deblur_tool,couter_frame_wode)#,strat_list=strat_list,index_1=index_)
                # # print('loss3june',loss3)
                # # True表示是相同的帧，false表示是不同的帧内容
                # if False == loss3:
                #     # 求第一帧的尾巴
                #     first_reserve,time_now = binary_find_the_same_frame(first_reserve, pre_image,contours_model_result[0],class_tool,deblur_tool,detectrou_tool,flag='tail',listsOfImg_position=listsOfImg_position,pre_time=pre_time,imageModel=imageModel,must_tail=first_200)
                #     # print('len(first_reserve)',len(first_reserve))
                #     delay = time_now - pre_time
                   
                #     if delay >= 200:
                #         dealy_ms[0] = dealy_ms[0] + delay
                #         print('200 pre_time is {},time_now is {}'.format(pre_time,time_now))

                #     if delay >= 300:
                #         dealy_ms[1] = dealy_ms[1] + delay
                #         print('300 pre_time is {},time_now is {}'.format(pre_time,time_now))

                #     if delay >= 600:
                #         print('600 pre_time is {},time_now is {}'.format(pre_time,time_now))
                #         dealy_ms[2] = dealy_ms[2] + delay
                #     pre_image = currentImage.copy()
                #     # 更新数组，找到和pre_image相同的帧，将其之前的帧都删除掉
                #     # print('second')
                #     # 求最后一帧的头
                #     first_reserve,time_now = binary_find_the_same_frame(first_reserve,currentImage,contours_model_result[0],class_tool,deblur_tool,detectrou_tool,flag='head',listsOfImg_position=listsOfImg_position,imageModel=imageModel,must_tail=(first_200 or last_200))
                #     pre_time = first_reserve[0][0]
                #     if first_200:
                #         first_200 = False
                #     # continue
                # else :# 如果是相同的,将当前数组全部删除，记录下当前的时间
                    # find_head_of_first_reserve = True
                    # print('相同')
                    # first_reserve = first_reserve[-1:]

            # listsOfImg, listsOfImg_position = pregetTimeresultOrDelay(current_frame, imageModel, flag_of_path,listsOfImg_position,detectrou_tool=detectrou_tool)
            # preImage, pre_time,pre_id, dealy_ms, counter_all ,index ,the_second_flag = toRecordDealy(listsOfImg, contours_model_result[0], thIfDelay,
            #                                                         preImage, pre_time,pre_id ,dealy_ms, counter_all,end_time,couter_frame_wode,class_tool,deblur_tool,strat_list = [] ,index_ =index,the_second_flag=the_second_flag)
            # # time.sleep(1)
            # couter_frame_wode = couter_frame_wode + 1
            # if preImage is None:
            #     break
                # istsOfImg_record = listsOfImg
                # listsOfImg_position_record = listsOfImg_position
            # result_list, list_confidence, name_flag, dst_path = getTimeresult(listsOfImg, deblur_tool, class_tool,
            #                                                                   contours_model_result, name_flag)
            # if len(result_list) >= 2:
            #     actual_result, result_of_time, flagOfTrueOrFalse = getResultRromList(result_list, list_confidence)
            #     print('actual_result', actual_result)
            #     if actual_result == 100000:
            #         listsOfImg_position = []
            #
            # if counter_frame > 1 and (counter_frame + 1) % 12 == 0:
            #     end_time_now = datetime.datetime.now()
            #     print('use time is {}'.format(end_time_now - begin_time_now))
            # counter_frame = counter_frame + 1

            
        
        # if pre_time != end_time:
        #         delay = end_time - pre_time
        #         # if delay >= 99:
        #         #
        #         #     # print('100 current_time is {},pre_time is {}'.format(current_time,pre_time))
        #         #     print('100 pre_id is {},current_id is {}'.format(pre_id,couter_frame_wode-1))
        #         #     print('100 current_time is {},pre_time is {}'.format(current_time,pre_time))
        #         #
        #         #     dealy_ms[3] = dealy_ms[3] + delay

                # if delay >= 200:
                #     print('200 pre_id is {},current_id is {}'.format(pre_id,couter_frame_wode-1))
                #     print('200 current_time is {},pre_time is {}'.format(current_time,pre_time))
                    
                #     dealy_ms[0] = dealy_ms[0] + delay
                # if delay >= 300:
                #     print('300 pre_id is {},current_id is {}'.format(pre_id,couter_frame_wode-1))
                #     print('300 current_time is {},pre_time is {}'.format(current_time,pre_time))
                    
                #     dealy_ms[1] = dealy_ms[1] + delay
                # if delay >= 600:
                #     print('600 current_time is {},pre_time is {}'.format(current_time,pre_time))
                #     print('600 pre_id is {},current_id is {}'.format(pre_id,couter_frame_wode-1))
                    
                #     dealy_ms[2] = dealy_ms[2] + delay