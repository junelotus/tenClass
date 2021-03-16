import numpy as np
import cv2
import os
import os.path
from pathlib import Path
import shutil
import sys


from test_deblur import DeBlur
from num_classification.test_classification  import Classification
from hough import *
def compator_x(elem):
    x = elem[0]
    y = elem[1]
    return x

def compator_y(elem):
    x = elem[0]
    y = elem[1]
    return y

def takeSecond(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return x

def takeY(cnt):
     x, y, w, h = cv2.boundingRect(cnt)
     return y

def takeAera(cnt):
     x, y, w, h = cv2.boundingRect(cnt)
     return w*h

def showImg(name,img):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name,img)
    cv2.waitKey(5000)

def getModelJEPGAnd15Positions(path):
    imageModel = cv2.imread(path)
    imageModel = cv2.cvtColor(imageModel, cv2.COLOR_RGB2GRAY)
    rct, imageModel = cv2.threshold(imageModel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours_model, _ = cv2.findContours(imageModel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_model.sort(key=takeSecond, reverse=False)
    contours_model.sort(key=takeY, reverse=False)
    contours_model_result = []
    for cnt in contours_model:
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        #print('area')
        aear = w1*h1
        #print(aear)
        img_curent_sub = imageModel[y1:y1+h1, x1:x1+w1]
        shape_sub = img_curent_sub.shape
        if not (shape_sub[0] < 30 or shape_sub[1] < 20 or shape_sub[0] >= 0.9*imageModel.shape[0] or
                shape_sub[1] >= 0.9*imageModel.shape[1]):
            contours_model_result.append(cnt)
    imageModel = imageModel/127.5-1
    return imageModel, contours_model_result

def L2Loss(img, model):
    if len(img.shape) == 3:
    
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rct, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if len(model.shape) == 3:
        model = cv2.cvtColor(model, cv2.COLOR_RGB2GRAY)
        rct, model = cv2.threshold(model, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # minx = min(img.shape[0],model.shape[0])
    # miny = min(img.shape[1],model.shape[1])
    shape = model.shape
    size = (shape[1], shape[0])
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    result = 0.0
    i = 0
    j = 0
    d = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            t = model[i][j] - img[i][j]
            # #print(t)
            result = result + t*t
    return result


def showImgByCnt(name, img, listsOfCnt):
    for x1, y1, w1, h1, L2Loss_result in listsOfCnt:   # i in range(1):#
        cv2.namedWindow('ori_'+name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('ori_'+name, img)
        cv2.namedWindow('current_'+name, cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
        cv2.imshow('current_'+name, img[y1:y1+h1, x1:x1+w1])
        #print(L2Loss_result)
        # cv2.waitKey(1000)
    

def findROI(image_color, imageModel, w_, h_, l2_loss_1, l2_loss_2, ratio):
    # img_color  = cv2.imread(p)
    img = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
    rct, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    shape = img.shape
    listsOfCnt = []
    x_before = -1
    y_before = -1
    w_before = -1
    h_before = -1
    contours.sort(key=takeSecond, reverse=False)
    for cnt in contours:
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        if w1*h1 < w1 < w_*h_ or w1 < w_ or h1 < h_:
            continue
        imgCurrent = img[y1:y1+h1, x1:x1+w1]

        imgCurrent_color = image_color[y1:y1+h1, x1:x1+w1]
        imgCurrent = imgCurrent/127.5-1
        L2Loss_result = L2Loss(imgCurrent, imageModel)
        if not (L2Loss_result > l2_loss_1 and  L2Loss_result < l2_loss_2) or \
                float(imgCurrent.shape[0])/float(imgCurrent.shape[1]) < ratio:
            continue
        if x_before != -1:
            while len(listsOfCnt) > 0 and (x1 >= listsOfCnt[-1][2] and y1 >= listsOfCnt[-1][3] and
                    x1 + w1 <= listsOfCnt[-1][2]+listsOfCnt[-1][0] and y1+h1 <= listsOfCnt[-1][3]+listsOfCnt[-1][1]):
                del(listsOfCnt[-1])
        if len(listsOfCnt) == 0:
                x_before = -1
                y_before = -1
                w_before = -1
                h_before = -1
                # continue
        listsOfCnt.append([ w1, h1, x1, y1,L2Loss_result])
        x_before = x1
        y_before = y1
        w_before = w1
        h_before = h1
    return listsOfCnt


def reShape(imageModel, imgCuerrnt):
        shape = imageModel.shape
        size = (shape[1], shape[0])
        imgCuerrnt = cv2.resize(imgCuerrnt, size, interpolation=cv2.INTER_LINEAR)
        return imgCuerrnt


def getRowWhiteCount(img, beginx, beginy, w, h):
    j = beginx
    counter = 0
    while j < min(beginx+h, img.shape[1]-1):
        if img[beginy][j] == 255:
            counter = counter+1
        j = j+1
    return counter


def getLieWhiteCount(img, beginx, beginy, w, h):
    i = beginy
    counter = 0
    while i < min(beginy+w, img.shape[0]-1):
        if img[i][beginx] == 255:
            counter = counter+1
        i = i+1
    return counter


def slideWindow(img, beginx, beginy, endx, endy, w, h, step):
    shape = img.shape
    oriy = beginy
    result_list = []
    beginx = max(0, beginx)
    beginy = max(0, beginy)
    #print(beginx, endx, beginy, endy)
    while beginx < endx:
        while beginy < endy:
            counter1 = getRowWhiteCount(img,beginx,beginy,w,h)
            counter2 = getRowWhiteCount(img,beginx,min(shape[0]-1,beginy+h),w,h)
            counter3 = getLieWhiteCount(img,beginx,beginy,w,h)
            counter4 = getLieWhiteCount(img,min(shape[1]-1,beginx+w),beginy,w,h)

            counter = counter1+counter2+counter3+counter4
            if counter ==  (2.0*(w+h)) :
                #print("shimashmashima")
                result_list.append([beginx,beginy,w,h])
                break

            beginy = beginy+step
        if len(result_list) > 0:
            break
        beginy = oriy
        beginx = beginx+step
    return result_list


def getAvaiableImg(p, listsOfCnt, w, h, step, imageModel):
    listsOfImg = []
    listsOfCnts = []
    angle_list = []
    img_color = cv2.imread(p)
    img = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    rct, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #print('test3')
    #print(len(listsOfCnt))
    shape = img_color.shape
    for w1, h1, x1, y1, L2Loss_result in listsOfCnt:
        imgCuerrnt = img[y1:y1+h1, x1:x1+w1]
        imgCuerrnt = reShape(imageModel, imgCuerrnt)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        imgCuerrnt = cv2.dilate(imgCuerrnt, kernel)
        imgCuerrnt = cv2.erode(imgCuerrnt, kernel)
        contours, hierarchy = cv2.findContours(imgCuerrnt, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours.sort(key=takeSecond, reverse=False)
        contours.sort(key=takeY, reverse=False)
        # if len(contours) < 15:
        #     continue
        # print('june')
        # print(x1,x1+w1,y1,y1+h1)
        img_result = img_color[y1:y1+h1, x1:x1+w1]
        y1 = max(y1-20,0)
        x1 = max(x1-20,0)
        y2 = min(y1+h1+40,img_color.shape[0]-1)#y1+h1#
        x2 = min(x1+w1+40,img_color.shape[1]-1)#x1+w1#
        img_to_hough = img_color[y1:y2, x1:x2]
        shape_of_hough = img_to_hough.shape
        # showImg('img_to_hough',img_to_hough)
        # print(img_color.shape)
        # print(shape_of_hough)
        # print(x1,x2,y1,y2)
        # if shape_of_hough[0] == 0 or shape_of_hough[1] == 0 or shape_of_hough[2] == 0 :
        #     continue
        angle_result = Hough(img_to_hough)
        print('**********junejunejune********************')
        print(angle_result)
        shape = img_result.shape
        matRotate = cv2.getRotationMatrix2D((shape[0]*0.5,shape[1]*0.5),angle_result,1)    # 为方便旋转后图片能完整展示，所以我们将其缩小
        img_result = cv2.warpAffine(img_result,matRotate,(shape[1],shape[0]))
        shape = img_result.shape
        if abs(angle_result) >= 5.0:
            # shape = img_result.shape
            # img_result = img_result[100:shape[0]-100, 100:shape[1]-100]
        # showImg('img_result',img_result)
            img_result_1 = cv2.cvtColor(img_result,cv2.COLOR_RGB2GRAY)
            rct, img_result_1 = cv2.threshold(img_result_1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # edges = cv2.Canny(img_result_1,0,255,apertureSize = 3)#cv2.Canny(img_result_1, CANNY_THRESH_1, CANNY_THRESH_2)
            contoursub, _ = cv2.findContours(img_result_1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            contoursub.sort(key=takeAera, reverse=True)
            if len(contoursub)>0:
                x,y,w,h = cv2.boundingRect(contoursub[0])
                img_result = img_result[y:y+h, x:x+w]
        # print(contours)
        # listsOfCnt = findROI(img_color, imageModel, 300, 300, 50000, 140000, 1)
        # showImg('img_result',img_result)
        listsOfImg.append(reShape(imageModel, img_result))#img_color[y1:y2, x1:x2]))
        listsOfCnts.append([w1, h1, x1, y1])#contours)
        angle_list.append(angle_result)

    return listsOfImg, listsOfCnts, angle_list


def setUtils(esc_encoder_wokl_2_path, ebc_encoder_wokl_2_path, eb_encoder_wokl_2_path, gen_s_wokl_2_path, tenClass_path):
    deblur_tool = DeBlur(1, esc_encoder_wokl_2_path, ebc_encoder_wokl_2_path, eb_encoder_wokl_2_path, gen_s_wokl_2_path)
    deblur_tool.set_buffer_size(1)
    class_tool = Classification(tenClass_path)
    class_tool.set_batch_size(15)
    return deblur_tool, class_tool


def getTimeresult(path, deblur_tool, class_tool, imageModel, contours_model_result,name_flag):
    python_path = os.environ["PYTHONPATH"].split(':')[0]
    to_save_dir = Path(python_path+'/Archive_to_split_2')
    if not to_save_dir.exists():
        os.makedirs(to_save_dir)
    to_save_sub_dir = Path(python_path+'/Archive_to_split_1')
    if not to_save_sub_dir.exists():
        os.makedirs(to_save_sub_dir)
    th = 0
    seconfOfPath = path.split('/')[-1]
    dst_path = []
    img_color = cv2.imread(path)#'/Users/xhj/opencv_bitbucket/Archive_to_split_2_last/446_PHOTO_IMG_0.jpg')#path)
    # showImg('img_color',img_color)
    # img_color_1 = cv2.imread('/Users/xhj/opencv_bitbucket/Archive_to_split_2_last/447_PHOTO_IMG_0.jpg')#path)
    listsOfCnt = findROI(img_color, imageModel, 300, 300, 50000, 140000, 1)
    listsOfImg, listsOfCnts, angle_list = getAvaiableImg(path, listsOfCnt, 54, 36, 5, imageModel)
    
    #listsOfImg 是hough变换之后的 listsOfCnts 是原始的坐标的
    # listsOfImg = []
    # listsOfImg.append(img_color)
    # listsOfImg.append(img_color_1)
    result_list_fiften_result = []
    list_confidence_fiften_result = []
    image_to_harr = None
    angle_to_harr = 0.0
    orb = cv2.ORB_create()
    for i in range(len(listsOfImg)):
        img  = listsOfImg[i]
        # img_to_harr_now = img_color[y1:y1+h1+40, x1-10:x1+w1+40]
        # print(listsOfCnts[i])
        w1, h1, x1, y1 = (listsOfCnts[i])
        # img_to_harr_now = img_color[y1:y1+h1,x1:x1+w1]
        img_to_harr_now = img_color[y1-10:y1+h1+40, x1-10:x1+w1+40]
        if img_to_harr_now.shape[0] == 0 or img_to_harr_now.shape[1] == 0 or img_to_harr_now.shape[2] == 0:
            img_to_harr_now = img
        # matRotate = cv2.getRotationMatrix2D((img_to_harr_now.shape[0]*0.5,img_to_harr_now.shape[1]*0.5),angle_list[i],1)    # 为方便旋转后图片能完整展示，所以我们将其缩小
        # img_to_harr_now = cv2.warpAffine(img_to_harr_now,matRotate,(img_to_harr_now.shape[1],img_to_harr_now.shape[0]))
            
        # showImg('img_to_harr',img_to_harr)
        pathToSave = str(to_save_dir)+'/'+str(name_flag)+'_'+seconfOfPath
        dst_path.append( pathToSave)
        cv2.imencode('.jpg', img_to_harr_now)[1].tofile(pathToSave)#img        

        result_list_fiften = []
        list_confidence_fiften = []
        a = 'a'
        img = reShape(imageModel, img)
        img_gery = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rct, img_binary = cv2.threshold(img_gery, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # showImg('ori', img)
        name_flag = name_flag+1
        for cnt in contours_model_result:
            x1, y1, w1, h1 = cv2.boundingRect(cnt)

            result_final = slideWindow(img_binary, x1-10, y1-10, x1+10, y1+10, w1+5, h1+5, 1)
            #print(result_final)
            #print(x1, y1, w1, h1)
            img_curent_sub1 = img[y1:y1+h1, x1:x1+w1]
            if len(result_final) > 0:
                x1 = result_final[0][0]
                y1 = result_final[0][1]
                w1 = result_final[0][2]
                h1 = result_final[0][3]
            #print(x1, y1, w1, h1)
            #print('area')
            aear = w1*h1
            #print(aear)
            img_curent_sub = img[y1:y1+h1,x1:x1+w1]
            shape_sub = img_curent_sub.shape
            if not (shape_sub[0] < 30 or  shape_sub[1] < 20 or shape_sub[0] >= 0.9*img.shape[0] or
                    shape_sub[1] >= 0.9*img.shape[1]):
                #print('junejune')
                #print(os.path.abspath('.'))
                pathToSave = str(to_save_sub_dir)+'/'+str(th)+'_'+seconfOfPath
                cv2.imencode('.jpg', img_curent_sub)[1].tofile(pathToSave)
                deblur_tool.set_path_to_deblur(pathToSave)
                deblur_tool.set_dateset()
                deblured_image = deblur_tool.to_get_result()

                pathToSave = str(to_save_sub_dir)+'/'+a+'_'+str(th)+'_'+seconfOfPath
                #print(pathToSave)
                img_gery = np.array(deblured_image[0]*0.5+0.5)*255
                cv2.imencode('.jpg', img_gery)[1].tofile(pathToSave)
                

                class_tool.set_path(pathToSave)

                result_list, list_confidence = class_tool.print_out()
                result_list_fiften.append(result_list[0])
                list_confidence_fiften.append(list_confidence[0])
                # os.remove(str(to_save_sub_dir)+'/'+str(th)+'_'+seconfOfPath)
                # os.remove(str(to_save_sub_dir)+'/'+a+'_'+str(th)+'_'+seconfOfPath)
                th = th+1
        result_list_fiften_result.append(result_list_fiften)
        list_confidence_fiften_result.append(list_confidence_fiften)
        print('***********************')
        print(result_list_fiften)
        if result_list_fiften.count(-1) <= 2 and image_to_harr == None:
            image_to_harr = img_to_harr_now
            angle_to_harr = angle_list[i]
            # showImg('image_to_harr',image_to_harr)
            break
    #print(listsOfCnt)
    # file_to_remove = glob.glob(str(to_save_dir)+'/*.jpg')
    # os.remove(file_to_remove)
    # file_to_remove = glob.glob(str(to_save_sub_dir)+'/*.jpg')
    # os.remove(file_to_remove)
    # shutil.rmtree(to_save_sub_dir, True)
    # shutil.rmtree(to_save_dir, True)
    return listsOfCnt, result_list_fiften_result, list_confidence_fiften_result,name_flag,dst_path,image_to_harr
def invalidate_index(result):
    result_invalidate_index = []
    shape = np.shape(result)
    for i in range(shape[0]):
        if result[i].count(-1) > 5:
            result_invalidate_index.append(i)
    return result_invalidate_index

# get the time on one picture,five nums
def getTheTime(sub_result_list,sub_list_confidence):
    result_one_time = []
    for i in range(5):
        nums_list = []
        nums_list.append(int(sub_result_list[i]))
        nums_list.append(int(sub_result_list[i + 5]))
        nums_list.append(int(sub_result_list[i + 10]))
        if nums_list.count(-1) >= 2:
            print('first pass')
            pass
        elif nums_list.count(-1) == 1:
            if nums_list[0] == -1 and (nums_list[1] + 1) % 10 == nums_list[2]:
                if nums_list[1] == 0:
                    result_one_time.append(9)
                else :
                    result_one_time.append(nums_list[1]-1)
            elif nums_list[1] == -1 and (nums_list[0]+2) % 10 == nums_list[2]:
                   result_one_time.append(nums_list[0])
            elif (nums_list[0]+1) % 10 == nums_list[1]:
                  result_one_time.append(nums_list[0])
        else :
            if (nums_list[0]+1) % 10 == nums_list[1] and (nums_list[1]+1) % 10 == nums_list[2] or (nums_list[0]+2) % 10 == nums_list[2] or (nums_list[0]+1) % 10 == nums_list[1] :
                result_one_time.append(nums_list[0])
            elif (nums_list[1] +1) % 10 == nums_list[2]:
                if nums_list[1] == 0:
                    result_one_time.append(9)
                else:
                    result_one_time.append(nums_list[1]-1)
        if len(result_one_time) == i:#!= i+1:
            confidence_result = getTimeByConfidence(sub_list_confidence,i)
            print('i is {},confidence_result is {}'.format(i,confidence_result))
            print('sub_list_confidence[i] is {}'.format(sub_list_confidence))
            if confidence_result == -1:
                return result_one_time
            else:
                result_one_time.append(confidence_result)
    return result_one_time

def getTimeByConfidence(sub_list_confidence,i):
    sub_list_confidence = np.delete(sub_list_confidence,0,axis=1)
    result_of_confidence = -1
    max_counter_of_index = 0
    max_confidence = 0.0
    counter  = 0
    while counter <= 9:
        print('counter is{}'.format(counter))
        if sub_list_confidence[i][counter] > 0.0 :
            print('a')
            sub_max_confidence = sub_list_confidence[i][counter]
            if sub_list_confidence[i+5][(counter+1) % 10] > 0.0:
                print('b')
                sub_max_confidence = sub_max_confidence+sub_list_confidence[i+5][(counter+1) % 10]
                if sub_max_confidence > max_confidence:
                    print('c')
                    max_confidence = sub_max_confidence
                    result_of_confidence = counter
                if sub_list_confidence[i+10][(counter+2) % 10] > 0.0:
                    print('d')
                    sub_max_confidence = sub_max_confidence+sub_list_confidence[i+10][(counter+2) % 10]
                    if sub_max_confidence > max_confidence:
                        print('e')
                        max_confidence = sub_max_confidence
                        result_of_confidence = counter
            elif sub_list_confidence[i+10][(counter+2) % 10] > 0.0:
                    print('f')
                    sub_max_confidence = sub_max_confidence+sub_list_confidence[i+10][(counter+2) % 10]
                    if sub_max_confidence > max_confidence:
                        print('g')
                        max_confidence = sub_max_confidence
                        result_of_confidence = counter
        else:
            print('h')
            sub_max_confidence = 0
            if sub_list_confidence[i+5][(counter+1) % 10] > 0.0  and sub_list_confidence[i+10][(counter+2) % 10] > 0.0:
                 print('i')
                 sub_max_confidence = sub_list_confidence[i+5][(counter+1) % 10]+sub_list_confidence[i+10][(counter+2) % 10]
                 if sub_max_confidence > max_confidence:
                        print('j')
                        max_confidence = sub_max_confidence
                        result_of_confidence = counter
        counter = counter+1
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
def is_LU(img_to_harr_gery,x,y):
    if img_to_harr_gery[x-1][y] < 100 and img_to_harr_gery[x-1][y-1] < 100 and img_to_harr_gery[x][y-1] < 100 and img_to_harr_gery[x+1][y-1] < 100 and img_to_harr_gery[x-1][y+1] < 100 and img_to_harr_gery[x][y+1] > 100 and img_to_harr_gery[x+1][y+1] > 100 and img_to_harr_gery[x+1][y] > 100:
        return True
    return False
def is_RU(img_to_harr_gery,x,y):
    if img_to_harr_gery[x-1][y] < 100 and img_to_harr_gery[x-1][y-1] < 100 and img_to_harr_gery[x][y-1] > 100 and img_to_harr_gery[x+1][y-1] > 100 and img_to_harr_gery[x-1][y+1] < 100 and img_to_harr_gery[x][y+1] < 100 and img_to_harr_gery[x+1][y+1] < 100 and img_to_harr_gery[x+1][y] > 100:
        return True
    return False
def is_LB(img_to_harr_gery,x,y):
    return False
def judege_circle_direction(begin,end,length,threshold):
    result_index = -1
    begin_percent = float(begin)/float(length)
    end_percent = float(end)/float(length)
    print('length',end+length-begin,end+length-begin/2)
    if begin > end and end+length-begin < length/3:
        # 根据两方的占比决定是1 or 4
        if length - begin > end :
            result_index = 1
        else:
            result_index = 2
    elif end_percent >= 0.25-threshold and end_percent <= 0.25+threshold and (begin == 0 or begin >= length*0.75):
        result_index = 2
    elif begin_percent >= 0.75-threshold and begin_percent <= 0.75+threshold and ( end <= length-1 and end >= length*0.75):
        result_index = 1
    elif begin_percent >= 0.5-threshold and begin_percent <= 0.5+threshold and end_percent >= 0.75-threshold and end_percent <= 0.75+threshold:
        result_index =  4
    elif begin_percent >= 0.25-threshold and begin_percent <= 0.25+threshold and end_percent >=0.5-threshold and end_percent <= 0.5+threshold:
        result_index = 3
    print('begin is {},end is {},length is{},result_index is {}'.format(begin,end,length,result_index))    
    return result_index
def construct_circle_to_judge_cornerdot(img_to_harr_binary,xx,yy,radius,shape_harr,thread_angle):
    pixel_list = []
    angle = 0.0
    begin_1 = False
    begin_0 = False
    change_1_0 = []
    change_0_1 = []
    white_counter = 0
    black_counter = 0
    if xx - radius < 0 or xx+radius >= shape_harr[0] or yy-radius < 0  or yy+radius >= shape_harr[1]:
        # print('first')
        return -1,-1,pixel_list,0,0
    while angle < 360.0:
        #anticlockwise，from Parallel to the x positive
        y = yy+int(radius*math.cos(math.radians(angle)))#(angle))#
        x = xx-int(radius*math.sin(math.radians(angle)))#(angle))#
        # print('x is {},y is {}'.format(x,y))
        if img_to_harr_binary[x][y] == 0:
            black_counter = black_counter+1
            pixel_list.append(0)
            if begin_0 == False and begin_1 == False:
                begin_0 = True
            if len(pixel_list) > 1 and pixel_list[-2] == 1 :
                change_1_0.append(len(pixel_list)-2)#记录下改变的位置
        else:
            pixel_list.append(1)
            white_counter = white_counter+1
            if begin_0 == False and begin_1 == False:
                begin_1 = True
            if len(pixel_list) > 1 and pixel_list[-2] == 0 :
                change_0_1.append(len(pixel_list)-2)#记录下改变的位置
        angle = angle+thread_angle
    # print('begin_0 is {},begin_1 is {},change_0_1 is {},change_1_0 is{}'.format(begin_0,begin_1,change_0_1,change_1_0)) 
    # 如果都只改变了一次
    if len(change_0_1) == 1 and len(change_1_0) == 1:
        if begin_0 :# 如果1全在中间
            print('second')
            return change_0_1[0]+1,change_1_0[0],pixel_list,white_counter,black_counter
        else: # 如果1是在首尾
            return change_0_1[0],change_1_0[0]+1,pixel_list,white_counter,black_counter
    elif len(change_0_1) == 1 and len(change_1_0) == 0:
            return change_0_1[0]+1,len(pixel_list)-1,pixel_list,white_counter,black_counter
    elif len(change_1_0) == 1 and len(change_0_1) == 0:
            return 0,change_1_0[0],pixel_list,white_counter,black_counter
    return -1,-1 ,pixel_list,white_counter,black_counter
#cornerdots_list 传入的list
#x_or_y ：挡墙一轮根据看x方向的离散程度和距离还是y方向的离散程度和距离
#coordinate_limit根据x和y方向，决定传入 0 或者shape[0] or shape[1]
def chose_final_corner_dot_list(cornerdots_list,x_or_y,coordinate_limit):
    if len(cornerdots_list ) < 4:
        return cornerdots_list
    mean  = [0,0] #2000000
    var = [2000000,2000000]
    result_list = []
    index_of_xory = 0
    # index_to_split_begin = 0
    # index_to_split_end = 0
    if x_or_y == 'x':
        cornerdots_list.sort(key=compator_x)#按照x排序
    else:
        cornerdots_list.sort(key=compator_y)#按照x排序
        index_of_xory = 1
    i = 2
    distance_of_xory = 2000000
    while i < len(cornerdots_list):
        mean_1 = (np.array(cornerdots_list[:i]).mean(axis=0))
        var_1 = (np.array(cornerdots_list[:i]).var(axis=0))
        # mean_2 = [0,0]
        # var_2 = sys.imtmax
        # if i == len(cornerdots_list)-1 :
        mean_2 = (np.array(cornerdots_list[i:]).mean(axis=0))
        var_2 = (np.array(cornerdots_list[i:]).var(axis=0))
        # 分别有上面两组，如果组内成都均值更加接近坐标极限值 并且 方差较小的化，选择此组
        
        distance_of_xory_1 = int(abs(mean_1[index_of_xory] - coordinate_limit))
        distance_of_xory_2 = int(abs(mean_2[index_of_xory] - coordinate_limit))
        if distance_of_xory_1 >= distance_of_xory_2:
            # 表示第二组的距离更接近 
            if distance_of_xory_2 < distance_of_xory  and var_2[index_of_xory] < var[index_of_xory]:
                # 如果第二组的距离更小并且方差更小，选择第二组
                result_list = cornerdots_list[i:]
                distance_of_xory = distance_of_xory_2
                var[index_of_xory] = var_2[index_of_xory]

        else :
            if distance_of_xory_1 < distance_of_xory  and var_1[index_of_xory] < var[index_of_xory]:
                # 如果第二组的距离更小并且方差更小，选择第二组
                result_list = cornerdots_list[:i]
                distance_of_xory = distance_of_xory_1
                var[index_of_xory] = var_1[index_of_xory]
        i = i+1
    return result_list

    # means = np.array(lu_list).mean(axis=0)
    # var = np.array(lu_list).var(axis=0)
    # print('********hhhhhhh',np.array(lu_list).mean(axis=0),np.array(lu_list).var(axis=0))
    # print('********hhhhhhh',np.array(lb_list).mean(axis=0),np.array(lb_list).var(axis=0))
    # print('********hhhhhhh',np.array(rb_list).mean(axis=0),np.array(rb_list).var(axis=0))
    # print('********hhhhhhh',np.array(ru_list).mean(axis=0),np.array(ru_list).var(axis=0))
def chose_final_corner_dot(list_1,list_2):
    list_2.sort(key=compator_x)
    length = min(len(list_1),len(list_2))
    index = 1
    for i in range(length):
        if list_1[i] != list_2[i]:
            index = i
            break
    return (np.array(lu_list[:index]).mean(axis=0))

def get_result_from_list(result_list,list_confidence):
    shape = np.shape(result_list)
    # #log.logger.infout('shape: {}', shape)
    flagOfTrueOrFalse = 0
    result_of_time = []
    result_invalidate_index = 0
    if shape[0] == 1:
        print('first')
        return 100000,result_of_time,flagOfTrueOrFalse
    if shape[0] >= 2:
        result_invalidate_index = invalidate_index(result_list)
        print('result_invalidate_index is {}'.format(result_invalidate_index))
        #log.logger.infout('result_invalidate_index: {}', result_invalidate_index)
    if shape[0] - len(result_invalidate_index) < 2:
        #log.logger.infout('result_invalidate_index0: {}', shape[0] - len(result_invalidate_index))
        print('second')
        return 100000,result_of_time,flagOfTrueOrFalse
    index_of_result_invalidate_index  = 0
    for i in range(shape[0]):
        if index_of_result_invalidate_index < len(result_invalidate_index) and \
                i == result_invalidate_index[index_of_result_invalidate_index]:
            index_of_result_invalidate_index = index_of_result_invalidate_index + 1
            continue
        else :
            print('result_list[i] is {}'.format(result_list[i]))
            result_time = getTheTime(result_list[i],list_confidence[i])
            print('result_time is {}'.format(result_time))
            print('i:{},result_time: {}'.format( i,result_time))
            #log.logger.infout('i:{},result_time: {}', i,result_time)
            if len(result_time) == 5:
                result_of_time.append(result_time)
    if len(result_of_time)!=2:
        #log.logger.infout('result_invalidate_index1: {}', result_of_time)
        print('third')
        return 100000,result_of_time,flagOfTrueOrFalse
    duliang = 10000
    first_num = 0
    second_num = 0
    #log.logger.infout('result_of_time: {}', result_of_time)
    for i in range(len(result_of_time)):
        flag  = True
        for j in range(len(result_of_time[0])-1):
           if (result_of_time[i][j]+1) % 10 != result_of_time[i][j+1]:
              flag  = False
              break
        if flag == True:
            flagOfTrueOrFalse = flagOfTrueOrFalse+1
    for i in range(len(result_of_time[0])):
        first_num = first_num+result_of_time[0][i]*duliang
        second_num = second_num+result_of_time[1][i]*duliang
        duliang = duliang/10
    result_final = (first_num + 100000  - second_num)%100000
    print(first_num,second_num)
    #log.logger.infout('first_num: {},second_num :{}', first_num,second_num)
    return  result_final,result_of_time,flagOfTrueOrFalse

python_path = os.environ["PYTHONPATH"].split(':')[0]
deblur_tool, class_tool = setUtils(
                    python_path+'/esc_encoder_wokl_3.h5',
                    python_path+'/ebc_encoder_wokl_3.h5',
                    python_path+'/eb_encoder_wokl_3.h5',
                    python_path+'/gen_s_wokl_3.h5',
                    python_path+'/tenClass_cpu.h5')

imageModel, contours_model_result = getModelJEPGAnd15Positions("model_8.jpg")
name_flag = 1
path_switch = glob.glob('./jpg18/*.jpg')
for path in path_switch:
    position, result_list, list_confidence,name_flag,dst_path,img_to_harr = getTimeresult(path, deblur_tool, class_tool, imageModel,contours_model_result,name_flag)
    if img_to_harr is None:
        continue
    #img_to_harr 是原始且没有翻转过的
    img_to_harr_gery = cv2.cvtColor(img_to_harr, cv2.COLOR_RGB2GRAY)
    rct, img_to_harr_binary = cv2.threshold(img_to_harr_gery, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    shape_harr = img_to_harr_binary.shape
    img_to_harr_gery_f32 = np.float32(img_to_harr_binary)#img_to_harr_gery)
    dst = cv2.cornerHarris(img_to_harr_gery_f32,2,3,0.04)
    dst = cv2.dilate(dst,None)
    #corrdia_list 为检测到的所有角点的位置
    corrdia_list = (np.array(np.argwhere((dst > 0.015*dst.max())==True))).tolist()#,dtype = dt)
    # img_to_harr[dst>0.01*dst.max()]=[0,0,255]
    # showImg('img_to_harr1',img_to_harr)
    # 将角点按照x和y坐标位置分别排序，分别取前图像前四分之一和四分之一的角点，
    #拿出来做环状扫描，得到开口方向和角度的，最终挑出4个角点做计算投影变换的变量
    corrdia_list.sort(key=compator_x)
    # corrdia_list.sort(key=compator_y)
    thread_to_harr_x = int(shape_harr[0]/15)
    thread_to_harr_y = int(shape_harr[1]/15)
    # corrdia_list_result =[]
    # lrub_list = [[]]
    lu_list = []
    lb_list = []
    rb_list = []
    ru_list = []

    for position in corrdia_list:
        x = position[0]
        y = position[1]
        if x <= thread_to_harr_x or x >= shape_harr[0] - thread_to_harr_x :
            begin,end ,pixel_list,white_counter,black_counter = construct_circle_to_judge_cornerdot(img_to_harr_binary,x,y,16,shape_harr,6)
            print('x is {},y is {},begin is {},end is {},white_counter is {},black_counter is {}'.format(x,y,begin,end ,white_counter,black_counter))
            if begin != -1:
                foreground_percent = float(white_counter)/float(black_counter+white_counter)
                if foreground_percent < 0.5 and foreground_percent > 0.2:
                    index = judege_circle_direction(begin,end,len(pixel_list),0.05)

                    print('*************hhhhh',x,y)
                    if index != -1:
                        # corrdia_list_result.append(position)
                        #img_to_harr[x][y] = [0,0,255]
                        if  index == 1:
                            #cv2.circle(img_to_harr,(y,x),16,(0,255,0))
                            lu_list.append(position)
                        if  index == 2:
                            #cv2.circle(img_to_harr,(y,x),16,(0,0,255))
                            lb_list.append(position)
                        if  index == 3:
                            #cv2.circle(img_to_harr,(y,x),16,(255,0,0))
                            rb_list.append(position)
                        if  index == 4:
                            #cv2.circle(img_to_harr,(y,x),16,(255,255,0))
                            ru_list.append(position)
        if  y <= thread_to_harr_y or y >= shape_harr[1] - thread_to_harr_y :
            begin,end ,pixel_list,white_counter,black_counter = construct_circle_to_judge_cornerdot(img_to_harr_binary,x,y,16,shape_harr,6)
            print('x is {},y is {},begin is {},end is {},white_counter is {},black_counter is {}'.format(x,y,begin,end ,white_counter,black_counter))
            if begin != -1:
                foreground_percent = float(white_counter)/float(black_counter+white_counter)
                if foreground_percent < 0.4 and foreground_percent > 0.2:
                    index = judege_circle_direction(begin,end,len(pixel_list),0.05)

                    print('*************hhhhh',x,y)
                    if index != -1:
                        # corrdia_list_result.append(position)
                        #img_to_harr[x][y] = [0,0,255]
                        if  index == 1:
                            #cv2.circle(img_to_harr,(y,x),16,(0,255,0))
                            lu_list.append(position)
                        if  index == 2:
                            #cv2.circle(img_to_harr,(y,x),16,(0,0,255))
                            lb_list.append(position)
                        if  index == 3:
                            #cv2.circle(img_to_harr,(y,x),16,(255,0,0))
                            rb_list.append(position)
                        if  index == 4:
                            #cv2.circle(img_to_harr,(y,x),16,(255,255,0))
                            ru_list.append(position)

    # lu_list.sort(key=compator_x)
    # lu_list = (set(lu_list))

    shape = img_to_harr.shape
    result_list_x = chose_final_corner_dot_list(lu_list,'x',0)
    print('************ddddddd',lu_list,result_list_x)
    # print(result_list_x)
    result_list_y = chose_final_corner_dot_list(lu_list,'y',0)
    print('************ddddddd',result_list_y)
    p1 = chose_final_corner_dot(result_list_x,result_list_y)
    print('************ttttt',p1)
    cv2.circle(img_to_harr,(int(result_list_y[0][1]),int(result_list_y[0][0])),16,(255,255,0))

    result_list_x = chose_final_corner_dot_list(lb_list,'x',shape[0])
    result_list_y = chose_final_corner_dot_list(lb_list,'y',0)
    p1 = chose_final_corner_dot(result_list_x,result_list_y)
    cv2.circle(img_to_harr,(int(result_list_y[0][1]),int(result_list_y[0][0])),16,(0,255,0))

    result_list_x = chose_final_corner_dot_list(rb_list,'x',shape[0])
    result_list_y = chose_final_corner_dot_list(rb_list,'y',shape[1])
    p1 = chose_final_corner_dot(result_list_x,result_list_y)
    cv2.circle(img_to_harr,(int(result_list_y[0][1]),int(result_list_y[0][0])),16,(255,0,0))

    result_list_x = chose_final_corner_dot_list(ru_list,'x',0)
    result_list_y = chose_final_corner_dot_list(ru_list,'y',shape[1])
    p1 = chose_final_corner_dot(result_list_x,result_list_y)
    cv2.circle(img_to_harr,(int(result_list_y[0][1]),int(result_list_y[0][0])),16,(0,0,255))


    # print('********hanjunne',len(corrdia_list_result))
    showImg('image_to_harr',img_to_harr)#img_to_harr)
    # shape = img_to_harr_gery.shape
    # print('***********3',dst.shape)
    # print('***********3',corrdia_list)
    # # Threshold for an optimal value, it may vary depending on the image.
    # img_to_harr[dst>0.01*dst.max()]=[0,0,255]#越小越多
    # # img_to_harr[corrdia_list[100][0],corrdia_list[100][1]] = [0,0,255]
    # #cv2.circle(img_to_harr,(corrdia_list[100][1],corrdia_list[100][0]),10,(0,255,0))
    
    # showImg('image_to_harr',img_to_harr_binary)#img_to_harr)
    cv2.imencode('.jpg', img_to_harr)[1].tofile('Archive_to_split_2/harr.jpg')#img        



# print(result_list)
# name_flag = name_flag+1
# if len(result_list) >= 2 :
#     actual_result,result_of_time,flagOfTrueOrFalse = get_result_from_list(result_list,list_confidence)
#     print(actual_result)
#     if actual_result == 100000:
#         pass
#         # counter_of_no_reconfig = counter_of_no_reconfig+1
#     else :
#         pass
#         # counter_of_yes_reconfig = counter_of_yes_reconfig+1
# else:
#     # counter_of_no_reconfig = counter_of_no_reconfig+1
#     actual_result = 0
# # print(actual_result)