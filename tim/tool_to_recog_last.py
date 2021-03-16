import numpy as np
import cv2
import os
import os.path
from pathlib import Path
import shutil


from test_deblur import DeBlur
from num_classification.test_classification  import Classification
from tool_perspectiv import *

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
    imageModel = cv2.imread(path)
    imageModel = cv2.cvtColor(imageModel, cv2.COLOR_RGB2GRAY)
    rct, imageModel = cv2.threshold(imageModel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours_model, _ = cv2.findContours(imageModel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_model.sort(key=takeSecond, reverse=False)
    contours_model.sort(key=takeY, reverse=False)
    contours_model_result = []
    for cnt in contours_model:
        x1, y1, w1, h1 = cv2.boundingRect(cnt)
        ###print('area')
        aear = w1*h1
        ###print(aear)
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
            # ###print(t)
            result = result + t*t
    return result

def showImgByCnt(name, img, listsOfCnt):
    for x1, y1, w1, h1, L2Loss_result in listsOfCnt:   # i in range(1):#
        cv2.namedWindow('ori_'+name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('ori_'+name, img)
        cv2.namedWindow('current_'+name, cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
        cv2.imshow('current_'+name, img[y1:y1+h1, x1:x1+w1])
        ###print(L2Loss_result)
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
    ###print(beginx, endx, beginy, endy)
    while beginx < endx:
        while beginy < endy:
            counter1 = getRowWhiteCount(img,beginx,beginy,w,h)
            counter2 = getRowWhiteCount(img,beginx,min(shape[0]-1,beginy+h),w,h)
            counter3 = getLieWhiteCount(img,beginx,beginy,w,h)
            counter4 = getLieWhiteCount(img,min(shape[1]-1,beginx+w),beginy,w,h)

            counter = counter1+counter2+counter3+counter4
            if counter ==  (2.0*(w+h)) :
                ###print("shimashmashima")
                result_list.append([beginx,beginy,w,h])
                break

            beginy = beginy+step
        if len(result_list) > 0:
            break
        beginy = oriy
        beginx = beginx+step
    return result_list

def getAvaiableImg(img_color, listsOfCnt, w, h, step, imageModel,flag_of_path):#(p, listsOfCnt, w, h, step, imageModel)
    listsOfImg = []
    listsOfCnts = []
    # img_color = cv2.imread(p)
    img = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    rct, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ###print('test3')
    ###print(len(listsOfCnt))
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
        # ##print('june')
        # ##print(x1,x1+w1,y1,y1+h1)

        # expand img_result to hough or perspective
        img_result = img_color[y1:y1+h1, x1:x1+w1]
        y1 = max(y1-20,0)# -20
        x1 = max(x1-20,0)
        y2 = min(y1+h1+40,img_color.shape[0]-1)#y1+h1#
        x2 = min(x1+w1+40,img_color.shape[1]-1)#x1+w1#
        img_to_correct = img_color[y1:y2, x1:x2]
        # 以下将hough变换改成仿射变换，如果没有找到四个角点，就使用hough
        # showImg('img_to_correct1',img_to_correct)
        img_result,solution = correctImageByPerspectiveOrHough(img_to_correct,img_result,flag_of_path)
        # to dont use classific
        
        if solution=='hough':
            continue
        flag_of_path = flag_of_path+1
        # shape_of_hough = img_to_hough.shape

        # angle_result = Hough(img_to_hough)
        # shape = img_result.shape
        # matRotate = cv2.getRotationMatrix2D((shape[0]*0.5,shape[1]*0.5),angle_result,1)    # 为方便旋转后图片能完整展示，所以我们将其缩小
        # img_result = cv2.warpAffine(img_result,matRotate,(shape[1],shape[0]))
        # shape = img_result.shape
        # if abs(angle_result) >= 5.0:
            # img_result_1 = cv2.cvtColor(img_result,cv2.COLOR_RGB2GRAY)
            # rct, img_result_1 = cv2.threshold(img_result_1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # contours, _ = cv2.findContours(img_result_1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            # contours.sort(key=takeAera, reverse=True)
            # if len(contours)>0:
            #     x,y,w,h = cv2.boundingRect(contours[0])
            #     img_result = img_result[y:y+h, x:x+w]
        listsOfImg.append([reShape(imageModel, img_result),solution])#img_color[y1:y2, x1:x2]))
        listsOfCnts.append(contours)

    return listsOfImg, listsOfCnts


def setUtils(esc_encoder_wokl_2_path, ebc_encoder_wokl_2_path, eb_encoder_wokl_2_path, gen_s_wokl_2_path, tenClass_path):
    deblur_tool = DeBlur(1, esc_encoder_wokl_2_path, ebc_encoder_wokl_2_path, eb_encoder_wokl_2_path, gen_s_wokl_2_path)
    deblur_tool.set_buffer_size(1)
    class_tool = Classification(tenClass_path)
    class_tool.set_batch_size(15)
    return deblur_tool, class_tool


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
            ##print('first pass')
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
            ##print('i is {},confidence_result is {}'.format(i,confidence_result))
            ##print('sub_list_confidence[i] is {}'.format(sub_list_confidence))
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
        ##print('counter is{}'.format(counter))
        if sub_list_confidence[i][counter] > 0.0 :
            ##print('a')
            sub_max_confidence = sub_list_confidence[i][counter]
            if sub_list_confidence[i+5][(counter+1) % 10] > 0.0:
                ##print('b')
                sub_max_confidence = sub_max_confidence+sub_list_confidence[i+5][(counter+1) % 10]
                if sub_max_confidence > max_confidence:
                    ##print('c')
                    max_confidence = sub_max_confidence
                    result_of_confidence = counter
                if sub_list_confidence[i+10][(counter+2) % 10] > 0.0:
                    ##print('d')
                    sub_max_confidence = sub_max_confidence+sub_list_confidence[i+10][(counter+2) % 10]
                    if sub_max_confidence > max_confidence:
                        ##print('e')
                        max_confidence = sub_max_confidence
                        result_of_confidence = counter
            elif sub_list_confidence[i+10][(counter+2) % 10] > 0.0:
                    ##print('f')
                    sub_max_confidence = sub_max_confidence+sub_list_confidence[i+10][(counter+2) % 10]
                    if sub_max_confidence > max_confidence:
                        ##print('g')
                        max_confidence = sub_max_confidence
                        result_of_confidence = counter
        else:
            ##print('h')
            sub_max_confidence = 0
            if sub_list_confidence[i+5][(counter+1) % 10] > 0.0  and sub_list_confidence[i+10][(counter+2) % 10] > 0.0:
                 ##print('i')
                 sub_max_confidence = sub_list_confidence[i+5][(counter+1) % 10]+sub_list_confidence[i+10][(counter+2) % 10]
                 if sub_max_confidence > max_confidence:
                        ##print('j')
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


def pregetTimeresultOrDelay(img_color, imageModel,flag_of_path):#papregetTimeresultOrDelay(path, imageModelth 
    # img_color = cv2.imread(path)
    listsOfCnt = findROI(img_color, imageModel, 300, 300, 50000, 140000, 1)
    listsOfImg, listsOfCnts = getAvaiableImg( img_color,listsOfCnt, 54, 36, 5, imageModel,flag_of_path)#path,
    return listsOfCnt,listsOfImg, listsOfCnts

def getTimeresult(listsOfImg, listsOfCnts,deblur_tool, class_tool, contours_model_result,name_flag):
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
    # img_color = cv2.imread(path)
    # listsOfCnt = findROI(img_color, imageModel, 300, 300, 50000, 140000, 1)
    # listsOfImg, listsOfCnts = getAvaiableImg(path, listsOfCnt, 54, 36, 5, imageModel)

    result_list_fiften_result = []
    list_confidence_fiften_result = []
    for imgzip in listsOfImg:
        img = imgzip[0]
        # if imgzip[1]=='hough':
        #     continue
        ##print('solution',imgzip[1])
        # pathToSave = str(to_save_dir)+'/'+str(name_flag)+'_'+seconfOfPath
        # dst_path.append( pathToSave)
        # cv2.imencode('.jpg', img)[1].tofile(pathToSave)
        
        # solution = imgzip[1]
        result_list_fiften = []
        list_confidence_fiften = []
        a = 'a'
        # img = reShape(imageModel, img)
        img_gery = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rct, img_binary = cv2.threshold(img_gery, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # showImg('ori', img)
        name_flag = name_flag+1
        for cnt in contours_model_result:
            x1, y1, w1, h1 = cv2.boundingRect(cnt)

            result_final = slideWindow(img_binary, x1-10, y1-10, x1+10, y1+10, w1+5, h1+5, 1)
            ###print(result_final)
            ###print(x1, y1, w1, h1)
            img_curent_sub1 = img[y1:y1+h1, x1:x1+w1]
            if len(result_final) > 0:
                x1 = result_final[0][0]
                y1 = result_final[0][1]
                w1 = result_final[0][2]
                h1 = result_final[0][3]
            ###print(x1, y1, w1, h1)
            ###print('area')
            aear = w1*h1
            ###print(aear)
            img_curent_sub = img[y1:y1+h1,x1:x1+w1]
            shape_sub = img_curent_sub.shape
            if not (shape_sub[0] < 30 or  shape_sub[1] < 20 or shape_sub[0] >= 0.9*img.shape[0] or
                    shape_sub[1] >= 0.9*img.shape[1]):
                ###print('junejune')
                ###print(os.path.abspath('.'))
                pathToSave = str(to_save_sub_dir)+'/'+str(th)+'_'+seconfOfPath
                cv2.imencode('.jpg', img_curent_sub)[1].tofile(pathToSave)
                #deblur ignor for delay
                deblur_tool.set_path_to_deblur(pathToSave)
                deblur_tool.set_dateset()
                deblured_image = deblur_tool.to_get_result()

                pathToSave = str(to_save_sub_dir)+'/'+a+'_'+str(th)+'_'+seconfOfPath
                ###print(pathToSave)
                img_gery = np.array(deblured_image[0]*0.5+0.5)*255
                cv2.imencode('.jpg', img_gery)[1].tofile(pathToSave)
                

                class_tool.set_path(pathToSave)

                result_list, list_confidence = class_tool.print_out()
                result_list_fiften.append(result_list[0])
                list_confidence_fiften.append(list_confidence[0])
                # os.remove(str(to_save_sub_dir)+'/'+str(th)+'_'+seconfOfPath)
                # os.remove(str(to_save_sub_dir)+'/'+a+'_'+str(th)+'_'+seconfOfPath)
                th = th+1
                # deblur ignor for delay
                # if len(result_list_fiften) == 3:
                #     break
        result_list_fiften_result.append(result_list_fiften)
        list_confidence_fiften_result.append(list_confidence_fiften)
        ###print('***********************')
        ###print(result_list_fiften)
    ###print(listsOfCnt)
    # file_to_remove = glob.glob(str(to_save_dir)+'/*.jpg')
    # os.remove(file_to_remove)
    # file_to_remove = glob.glob(str(to_save_sub_dir)+'/*.jpg')
    # os.remove(file_to_remove)
    # shutil.rmtree(to_save_sub_dir, True)
    # shutil.rmtree(to_save_dir, True)
    return  result_list_fiften_result, list_confidence_fiften_result,name_flag,dst_path

def getDelayResultImage(listsOfImg,cnt):
    i = len(listsOfImg)-1
    while i >= 0:
        if listsOfImg[i][1] == 'perspective':# and (result_list[i].count(-1) == 0):
            img = listsOfImg[i][0]
            img_gery = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            rct, img_binary = cv2.threshold(img_gery, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
         
            x1, y1, w1, h1 = cv2.boundingRect(cnt)
            result_final = slideWindow(img_binary, x1-10, y1-10, x1+10, y1+10, w1+5, h1+5, 1)
            ###print(result_final)
            ###print(x1, y1, w1, h1)
            # img_curent_sub1 = img[y1:y1+h1, x1:x1+w1]
            if len(result_final) > 0:
                x1 = result_final[0][0]
                y1 = result_final[0][1]
                w1 = result_final[0][2]
                h1 = result_final[0][3]
            img_curent_sub = img_binary[y1:y1+h1,x1:x1+w1]
            counter_white = 0
            for x in range( img_curent_sub.shape[0]):
                for y in range(img_curent_sub.shape[1]):
                    if img_curent_sub[x][y]  > 0:
                        counter_white = counter_white+1
            if counter_white /(img_curent_sub.shape[0]*img_curent_sub.shape[1]) < 0.7:
                return listsOfImg[i][0]
        i = i-1
    return None
def get_result_from_list(result_list,list_confidence):
    shape = np.shape(result_list)
    # #log.logger.infout('shape: {}', shape)
    flagOfTrueOrFalse = 0
    result_of_time = []
    result_invalidate_index = 0
    if shape[0] == 1:
        # ##print('first')
        return 100000,result_of_time,flagOfTrueOrFalse
    if shape[0] >= 2:
        result_invalidate_index = invalidate_index(result_list)
        # ##print('result_invalidate_index is {}'.format(result_invalidate_index))
        #log.logger.infout('result_invalidate_index: {}', result_invalidate_index)
    if shape[0] - len(result_invalidate_index) < 2:
        #log.logger.infout('result_invalidate_index0: {}', shape[0] - len(result_invalidate_index))
        # ##print('second')
        return 100000,result_of_time,flagOfTrueOrFalse
    index_of_result_invalidate_index  = 0
    for i in range(shape[0]):
        if index_of_result_invalidate_index < len(result_invalidate_index) and \
                i == result_invalidate_index[index_of_result_invalidate_index]:
            index_of_result_invalidate_index = index_of_result_invalidate_index + 1
            continue
        else :
            ##print('result_list[i] is {}'.format(result_list[i]))
            result_time = getTheTime(result_list[i],list_confidence[i])
            ##print('result_time is {}'.format(result_time))
            ##print('i:{},result_time: {}'.format( i,result_time))
            #log.logger.infout('i:{},result_time: {}', i,result_time)
            if len(result_time) == 5:
                result_of_time.append(result_time)
    if len(result_of_time)!=2:
        #log.logger.infout('result_invalidate_index1: {}', result_of_time)
        ##print('third')
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
    print('ccccccc',first_num,second_num)
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
# path_all = glob.glob('/Users/xhj/Downloads/VID20200814162409.mp4')
# for path in path_all:
capture = cv2.VideoCapture('/Users/xhj/Downloads/VID20200814162409.mp4')#'/Users/xhj/Downloads/162050.mp4')#
name_flag = 0
flag_of_path = 0
preImage = cv2.resize(imageModel,(32,32))
path_of_all = glob.glob('PHOTO_IMG_153630_0.jpg')#/Users/xhj/Downloads/162050/*jpg')#'PHOTO_IMG_153630_0.jpg')
counter_all = 0
counter_delay = 0
first_delay_flag = True
pre_time = 0.0
current_time = 0.0
dealy_200ms_counter = 0
dealy_200ms = 0.0
dealy_300ms_counter = 0
dealy_300ms = 0.0
dealy_600ms_counter = 0
dealy_600ms = 0.0
begin_time = 0.0
end_time = 0.0
while capture.isOpened():
    frame_exists, current_frame = capture.read()
    if  not frame_exists:
        break
# for p in path_of_all:
#     #print(p)
#     current_frame = cv2.imread(p)
    position,listsOfImg, listsOfCnts = pregetTimeresultOrDelay(current_frame, imageModel,flag_of_path)
    # result_list, list_confidence,name_flag,dst_path = getTimeresult(listsOfImg, listsOfCnts, deblur_tool, class_tool,contours_model_result,name_flag)
    currentImage = getDelayResultImage(listsOfImg,contours_model_result[0])
    # preImage = cv2.resize(preImage, (32, 32))
    if currentImage is None :
        continue
    currentImage = cv2.resize(currentImage, (32, 32))
    # showImg('preImage',preImage)
    loss = L2Loss(preImage,currentImage)
    counter_all = counter_all+1
    if first_delay_flag == False:
        
        end_time = int(capture.get(cv2.CAP_PROP_POS_MSEC))
        current_time = end_time
        #print('current_time is {},pre_time is {}'.format(current_time,pre_time))
        if loss > 20:
            
            #int(capture.get(cv2.CAP_PROP_POS_MSEC))
        # else:
            delay = current_time - pre_time
            if delay >= 200:
                dealy_200ms_counter = dealy_200ms_counter+1
                dealy_200ms = dealy_200ms + delay
            if delay >= 300:
                dealy_300ms_counter = dealy_300ms_counter+1
                dealy_300ms = dealy_300ms + delay
            if delay >= 600:
                dealy_600ms_counter = dealy_600ms_counter+1
                dealy_600ms = dealy_600ms + delay
            pre_time = end_time #int(capture.get(cv2.CAP_PROP_POS_MSEC))

    if first_delay_flag:
        first_delay_flag = False
        pre_time = int(capture.get(cv2.CAP_PROP_POS_MSEC))
        begin_time = pre_time
    print(counter_all)

    # print('dealy_200ms_counter is {}'.format(dealy_200ms_counter))
    # print('dealy_300ms_counter is {}'.format(dealy_300ms_counter))
    # print('dealy_600ms_counter is {}'.format(dealy_600ms_counter))
    # print('counter_all is {}'.format(counter_all))

    # print('dealy_200ms is {}'.format(dealy_200ms))
    # print('dealy_300ms is {}'.format(dealy_300ms))
    # print('dealy_600ms is {}'.format(dealy_600ms))
    # print('time_all is {}'.format(current_time))
    # print('eeeeeee',loss)
    preImage = currentImage.copy()
    # preImage = cv2.resize(preImage,(32,32))
    flag_of_path = flag_of_path+3
    #print('aaaaaa',result_list)
    # ignor
    # if len(result_list) >= 2 :
    #     actual_result,result_of_time,flagOfTrueOrFalse = get_result_from_list(result_list,list_confidence)
    #     #print('bbbbb',actual_result)
    #     if actual_result == 100000:
    #         pass
    #         # counter_of_no_reconfig = counter_of_no_reconfig+1
    #     else :
    #         pass
    #         # counter_of_yes_reconfig = counter_of_yes_reconfig+1
    # else:
    #     # counter_of_no_reconfig = counter_of_no_reconfig+1
    #     actual_result = 0
    ##print(actual_result)
time_all = end_time-begin_time
#print('delay 200ms is {}'.format(dealy_200ms_counter/time_all))
#print('delay 300ms is {}'.format(dealy_300ms_counter/time_all))
#print('delay 600ms is {}'.format(dealy_600ms_counter/time_all))
    