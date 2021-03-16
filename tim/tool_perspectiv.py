import numpy as np
import cv2
import os
import os.path
from pathlib import Path
from hough import *
import shutil

def compator_x(elem):
    x = elem[0]
    y = elem[1]
    return x

def compator_y(elem):
    x = elem[0]
    y = elem[1]
    return y

def takeAera(cnt):
     x, y, w, h = cv2.boundingRect(cnt)
     return w*h

def thresholdImage(img_to_correct):
    img_result_1 = cv2.cvtColor(img_to_correct,cv2.COLOR_RGB2GRAY)
    rct, img_result_1 = cv2.threshold(img_result_1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img_result_1
    
def rodataImageToCorrect(img_to_correct):
    img_result_1 = thresholdImage(img_to_correct)
    # #showImg('gery',img_result_1)
    contoursub, _ = cv2.findContours(img_result_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contoursub.sort(key=takeAera, reverse=True)
    # find the minrect,to correct it if necessary 
    min_rect = cv2.minAreaRect(contoursub[0])
    # ##print(min_rect)
    angle = min_rect[2]
    if abs(min_rect[2]) >= 45:
        angle = 90-abs(min_rect[2])
    matRotate = cv2.getRotationMatrix2D((img_to_correct.shape[0]*0.5,img_to_correct.shape[1]*0.5),angle,1)    # 为方便旋转后图片能完整展示，所以我们将其缩小
    img_to_correct = cv2.warpAffine(img_to_correct,matRotate,(img_to_correct.shape[1],img_to_correct.shape[0]))
    # #showImg('rotated',img_to_correct)
    return img_to_correct

def constructCircleToJudgeCornerDot(img_to_correct_binary,xx,yy,radius,shape_harr,thread_angle):
    pixel_list = []
    angle = 0.0
    begin_1 = False
    begin_0 = False
    change_1_0 = []
    change_0_1 = []
    white_counter = 0
    black_counter = 0
    if xx - radius < 0 or xx+radius >= shape_harr[0] or yy-radius < 0  or yy+radius >= shape_harr[1]:
        ##print('first')
        return -1,-1,pixel_list,0,0
    while angle < 360.0:
        #anticlockwise，from Parallel to the x positive
        y = yy+int(radius*math.cos(math.radians(angle)))#(angle))#
        x = xx-int(radius*math.sin(math.radians(angle)))#(angle))#
        # ##print('x is {},y is {}'.format(x,y))
        if img_to_correct_binary[x][y] == 0:
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
    ##print('begin_0 is {},begin_1 is {},change_0_1 is {},change_1_0 is{}'.format(begin_0,begin_1,change_0_1,change_1_0)) 
    # 如果都只改变了一次
    if len(change_0_1) == 1 and len(change_1_0) == 1:
        if begin_0 :# 如果1全在中间
            # ##print('second')
            return change_0_1[0]+1,change_1_0[0],pixel_list,white_counter,black_counter
        else: # 如果1是在首尾
            return change_0_1[0],change_1_0[0]+1,pixel_list,white_counter,black_counter
    elif len(change_0_1) == 1 and len(change_1_0) == 0:
            return change_0_1[0]+1,len(pixel_list)-1,pixel_list,white_counter,black_counter
    elif len(change_1_0) == 1 and len(change_0_1) == 0:
            return 0,change_1_0[0],pixel_list,white_counter,black_counter
    return -1,-1 ,pixel_list,white_counter,black_counter
def choseFinalFornerDotList(cornerdots_list,x_or_y,coordinate_limit_1,coordinate_limit_2,thread_var):
    mean = (np.array(cornerdots_list).mean(axis=0))
    var = (np.array(cornerdots_list).var(axis=0))
    if var[0] < thread_var and var[1] <= thread_var : # 行和列的方差都比较小
        return cornerdots_list,False,mean
    # if len(cornerdots_list ) < 4:
    #     return cornerdots_list
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
    j = 0
    distance_of_xory = 200000000
    while j < len(cornerdots_list)-1 :
        i = j+1
        while i < len(cornerdots_list):
            mean_1 = (np.array(cornerdots_list[j:i+1]).mean(axis=0))
            var_1 = (np.array(cornerdots_list[j:i+1]).var(axis=0))
            distance_of_xory_1 = int(abs(mean_1[0] - coordinate_limit_1) * abs(mean_1[1] - coordinate_limit_2))
            # ##print("****yyyyy",cornerdots_list[j:i+1])
            # ##print("****ggggg",distance_of_xory_1,var_1,var)
            if distance_of_xory_1 < distance_of_xory and var_1[0] < thread_var and var_1[1] < thread_var:
                mean = mean_1
                result_list = cornerdots_list[j:i+1]
                distance_of_xory = distance_of_xory_1
                var[index_of_xory] = var_1[index_of_xory]
            i = i+1
        j = j+1
    return result_list,True,mean

    # means = np.array(lu_list).mean(axis=0)
    # var = np.array(lu_list).var(axis=0)
    # ##print('********hhhhhhh',np.array(lu_list).mean(axis=0),np.array(lu_list).var(axis=0))
    # ##print('********hhhhhhh',np.array(lb_list).mean(axis=0),np.array(lb_list).var(axis=0))
    # ##print('********hhhhhhh',np.array(rb_list).mean(axis=0),np.array(rb_list).var(axis=0))
    # ##print('********hhhhhhh',np.array(ru_list).mean(axis=0),np.array(ru_list).var(axis=0))
def choseFinalFornerDot(list_1,list_2):
    # list_2.sort(key=compator_x)
    # length = min(len(list_1),len(list_2))
    # index = 1
    # for i in range(length):
    #     if list_1[i] != list_2[i]:
    #         index = i
    #         break
    return 0.5*((np.array(list_1).mean(axis=0))+(np.array(list_2).mean(axis=0)))


def drawPictureUseCornerDotOri(AffineImg,img_to_correct,AffineMatrix,width,height,canvas_point):
    u_k = (canvas_point[3][1]-canvas_point[0][1])/(canvas_point[3][0]-canvas_point[0][0])
    u_d = canvas_point[3][1] - u_k*canvas_point[3][0]
    
    b_k = (canvas_point[2][1]-canvas_point[1][1])/(canvas_point[2][0]-canvas_point[1][0])
    b_d = canvas_point[2][1] - b_k*canvas_point[2][0]
    
    l_k = (canvas_point[1][1]-canvas_point[0][1])/(canvas_point[1][0]-canvas_point[0][0])
    l_d = canvas_point[0][1] - l_k*canvas_point[0][0]

    r_k =  (canvas_point[2][1]-canvas_point[3][1])/(canvas_point[2][0]-canvas_point[3][0])
    r_d = canvas_point[2][1] - r_k*canvas_point[2][0]
    # ##print('endendendend')
    # ##print(u_k,b_k,l_k,r_k)
    result_perspective =  np.ones((height,width,3),dtype=np.uint8)#img_to_correct[0:height,0:width]
    # cv2.circle(img_to_correct, ( 0,int(img_to_correct.shape[0]-10)), 3, (0, 255, 0), 30)
    # #showImg('test',img_to_correct)
    
    # ##print(np.dot(AffineMatrix,[canvas_point[3][0],canvas_point[3][1],1]))
    for x in range(img_to_correct.shape[0]):
        for y in range(img_to_correct.shape[1]):
            if ( ((u_k >= 0 and y-u_k*x-u_d  <= 0.0 ) or ( u_k< 0 and  y-u_k*x-u_d  >= -1.0))
             and  ( (b_k >=0 and  y-b_k*x-b_d  >= 0.0) or ( b_k <=0  and y-b_k*x-b_d  <= -1.0)) 
            and  y-l_k*x-l_d  >= 1.0
             and y-r_k*x-r_d <= -1.0):
                src = np.array([x,y,1])
                # ##print('nownownownow')
                # ##print(src)
                dst = np.dot(AffineMatrix,src)
                dst_x = int(dst[0]/dst[-1])
                dst_y = int(dst[1]/dst[-1])
                if dst_x >= 0 and dst_x < result_perspective.shape[0] and dst_y >=0 and dst_y < result_perspective.shape[1]:
                    result_perspective[dst_x][dst_y] = img_to_correct[x][y]

    return result_perspective


def judegeCircleDirection(begin,end,length,threshold):
    result_index = -1
    # if  begin > end and end+length-begin < length/2.5 :
    #     first_part  = begin+1
    #     second_part = length-1-end
    #     if first_part > second_part:
    #         result_index = 2
    #     else :
    #         result_index = 1
    # elif (end-begin+1) < length/2:
    #     if begin >= 0 and begin < length/4:
    #         if end >= 0 and end < length/4:
    #             result_index =  2
    #         elif end > length/4  and end <= length/2:
    #             first_part  = length/4  - begin
        #         second_part = end -length/4+1
        #         if  first_part > second_part:
        #             result_index = 2
        #         else:
        #             result_index = 3
        # elif begin >= length/4 and begin < length/2:
        #     if end >= length/4 and end < length/2:
        #         result_index = 3
        #     elif end > length/2  and end <= length/4*3:
        #         first_part  = length/2  - begin
        #         second_part = end -length/2+1
        #         if  first_part > second_part:
        #             result_index = 3
        #         else:
        #             result_index = 4
        # elif begin >= length/2 and begin < length/4*3:
        #     if end > length/2 and end <= length/4*3:
        #         result_index = 4
        #     elif end >= length/4*3  and end < length:
        #         first_part  = length/4*3  - begin
        #         second_part = end -length/4*3+1
        #         if  first_part > second_part:
        #             result_index = 4
        #         else:
        #             result_index = 1
        # else:
        #     if  begin >= length/4*3 and begin < length and end > length/4*3 and end <= length:
        #         result_index = 1
    begin_percent = float(begin)/float(length)
    end_percent = float(end)/float(length)
    # ##print('length',end+length-begin,end+length-begin/2)
    if begin > end and end+length-begin < length/2.5:
        # 根据两方的占比决定是1 or 2
        if (length - begin) / end > 2.0  :
            result_index = 1
        elif  end/ (length - begin) > 2.0:
            result_index = 2
    elif end_percent >= 0.25-threshold and end_percent <= 0.25+threshold and ((begin >= 0 and begin_percent < threshold) or begin >= length*0.85):
        result_index = 2
    elif begin_percent >= 0.75-threshold and begin_percent <= 0.75+threshold and ( (end <= length-1 and end >= length*0.75) or end <= length*threshold):
        result_index = 1
    elif begin_percent >= 0.5-threshold and begin_percent <= 0.5+threshold and end_percent >= 0.75-threshold and end_percent <= 0.75+threshold:
        result_index = 4
    elif begin_percent >= 0.25-threshold and begin_percent <= 0.25+threshold and end_percent >=0.5-threshold and end_percent <= 0.5+threshold:
        result_index = 3
    # #print('begin is {},end is {},length is{},result_index is {}'.format(begin,end,length,result_index))    
    return result_index

def correctImageByPerspectiveUseCnt(img, cnt, width, height):
    # 获取包围contour的多边形, 应该是个4边形
    EPSILON = 5
    polygon = cv2.approxPolyDP(cnt, EPSILON, True)[:, 0]
    if len(polygon) != 4:
        # #print('firsttoreturn')
        return polygon, None

    # 获取多边形的4个顶点
    center = np.int0(cv2.minAreaRect(cnt)[0])
    X, Y = 0, 1  # 定义下面结构中X和Y的索引
    top_left_x = np.min(polygon[:, X][np.where(polygon[:, Y] < center[Y])])
    top_left_y = np.min(polygon[:, Y][np.where(polygon[:, X] == top_left_x)])
    bottom_left_x = np.min(polygon[:, X][np.where(polygon[:, Y] > center[Y])])
    bottom_left_y = np.max(polygon[:, Y][np.where(polygon[:, X] == bottom_left_x)])
    top_right_x = np.max(polygon[:, X][np.where(polygon[:, Y] < center[Y])])
    top_right_y = np.min(polygon[:, Y][np.where(polygon[:, X] == top_right_x)])
    bottom_right_x = np.max(polygon[:, X][np.where(polygon[:, Y] > center[Y])])
    bottom_right_y = np.max(polygon[:, Y][np.where(polygon[:, X] == bottom_right_x)])

    # 生成透视转换矩阵
    (x, y, w, h) = cv2.boundingRect(cnt)
    quad_pts = np.array([[top_left_x - x, top_left_y - y],
                         [bottom_left_x - x, bottom_left_y - y],
                         [top_right_x - x, top_right_y - y],
                         [bottom_right_x - x, bottom_right_y - y]],
                        np.float32)
    squre_pts = np.array([[0, 0],
                          [0, height],
                          [width, 0],
                          [width, height]],
                         np.float32)
    transform = cv2.getPerspectiveTransform(quad_pts, squre_pts)

    # 进行透视转换
    img_roi = img[y:y + h, x:x + w]
    # #showImg('second',img_roi)
    img_roi = cv2.warpPerspective(img_roi, transform, (width, height))

    # 返回4个角点的数组和矫正后的image
    return ((top_left_x, top_left_y),
            (bottom_left_x, bottom_left_y),
            (top_right_x, top_right_y),
            (bottom_right_x, bottom_right_y)
            ), img_roi


def correctImageByPerspectiveUseCornerOrHough(img_to_correct,img_result,flag_of_path = 0):
    # img_to_correct = rodataImageToCorrect(img_to_correct)

    # img_to_correct_gery = cv2.cvtColor(img_to_correct, cv2.COLOR_RGB2GRAY)
    # rct, img_to_correct_binary = cv2.threshold(img_to_correct_gery, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # to find all corner dots
    img_to_correct_binary = thresholdImage(img_to_correct)
    shape_harr = img_to_correct_binary.shape
    img_to_correct_gery_f32 = np.float32(img_to_correct_binary)#img_to_correct_gery)
    dst = cv2.cornerHarris(img_to_correct_gery_f32,2,3,0.04)
    dst = cv2.dilate(dst,None)
    #showImg('img_to_correct_gery_f32',img_to_correct_gery_f32)
    corrdia_list = (np.array(np.argwhere((dst > 0.015*dst.max())==True))).tolist()#,dtype = dt)
    img_to_correct1 = img_to_correct.copy()
    img_to_correct1[dst>0.01*dst.max()]=[0,0,255]
    #showImg('img_to_harr1',img_to_correct1)
    # img_to_correct[dst>0.01*dst.max()]=[0,0,255]
    # #showImg('img_to_correct1',img_to_correct)
    # 将角点按照x和y坐标位置分别排序，分别取前图像前四分之一和四分之一的角点，
    #拿出来做环状扫描，得到开口方向和角度的，最终挑出4个角点做计算投影变换的变量
    corrdia_list.sort(key=compator_x)
    # corrdia_list.sort(key=compator_y)
    threshold_to_harr_x = int(shape_harr[0]/8)
    threshold_to_harr_y = int(shape_harr[1]/8)
    # corrdia_list_result =[]
    # lrub_list = [[]]
    lu_list = []
    lb_list = []
    rb_list = []
    ru_list = []
    canvas_point = []
    for position in corrdia_list:
        x = position[0]
        y = position[1]
        # set x threshold to screen the dot
        if x <= threshold_to_harr_x or x >= shape_harr[0] - threshold_to_harr_x :
            begin,end ,pixel_list,white_counter,black_counter = constructCircleToJudgeCornerDot(img_to_correct_binary,x,y,14,shape_harr,6)
            # #print('x is {},y is {},begin is {},end is {},white_counter is {},black_counter is {}'.format(x,y,begin,end ,white_counter,black_counter))
            if begin != -1:
                foreground_percent = float(white_counter)/float(black_counter+white_counter)
                if foreground_percent < 0.4 and foreground_percent > 0.2:
                    index = judegeCircleDirection(begin,end,len(pixel_list),0.08)
                    # if index == -1:
                    #     cv2.circle(img_to_correct,(y,x),16,(0,255,255))
                    # ##print('*************hhhhh',x,y)
                    if index != -1:
                        # corrdia_list_result.append(position)
                        #img_to_correct[x][y] = [0,0,255]
                        if  index == 1:
                            ##cv2.circle(img_to_correct,(y,x),16,(0,255,0))
                            if lu_list.count(position) == 0:
                                lu_list.append(position)
                        if  index == 2:
                            ##cv2.circle(img_to_correct,(y,x),16,(0,0,255))
                            if lb_list.count(position) == 0:
                                lb_list.append(position)
                        if  index == 3:
                            ##cv2.circle(img_to_correct,(y,x),16,(255,0,0))
                            if rb_list.count(position) == 0:
                                rb_list.append(position)
                        if  index == 4:
                            ##cv2.circle(img_to_correct,(y,x),16,(255,255,0))
                            if ru_list.count(position) == 0:
                                ru_list.append(position)
        # set y threshold to screen corner dots
        if  y <= threshold_to_harr_y or y >= shape_harr[1] - threshold_to_harr_y :
            begin,end ,pixel_list,white_counter,black_counter = constructCircleToJudgeCornerDot(img_to_correct_binary,x,y,16,shape_harr,6)
            # ##print('x is {},y is {},begin is {},end is {},white_counter is {},black_counter is {}'.format(x,y,begin,end ,white_counter,black_counter))
            if begin != -1:
                foreground_percent = float(white_counter)/float(black_counter+white_counter)
                if foreground_percent < 0.4 and foreground_percent > 0.2:
                    index = judegeCircleDirection(begin,end,len(pixel_list),0.05)
                    # if index == -1:
                    #     cv2.circle(img_to_correct,(y,x),16,(255,0,255))
                    # ##print('*************hhhhh',x,y)
                    if index != -1:
                        if  index == 1:
                            ##cv2.circle(img_to_correct,(y,x),16,(0,255,0))
                            if lu_list.count(position) == 0:
                                lu_list.append(position)
                        if  index == 2:
                            ##cv2.circle(img_to_correct,(y,x),16,(0,0,255))
                            if lb_list.count(position) == 0:
                                lb_list.append(position)
                        if  index == 3:
                            ##cv2.circle(img_to_correct,(y,x),16,(255,0,0))
                            if rb_list.count(position) == 0:
                                rb_list.append(position)
                        if  index == 4:
                            ##cv2.circle(img_to_correct,(y,x),16,(255,255,0))
                            if ru_list.count(position) == 0:
                                ru_list.append(position)

    # lu_list.sort(key=compator_x)
    # lu_list = list(set(lu_list))

    shape = img_to_correct.shape
    if len(lu_list) > 0 :
        # #print('************aaaaaa',lu_list)
        result_list_x,need_y,p1 = choseFinalFornerDotList(lu_list,'x',0,0,10)
        if need_y :
            result_list_y,_,p1 = choseFinalFornerDotList(lu_list,'y',0,0,10)
            p1 = choseFinalFornerDot(result_list_x,result_list_y)
        if p1.size == 2 and p1[0]!=np.nan and p1[1]!=np.nan:    
            cv2.circle(img_to_correct,(int(p1[1]),int(p1[0])),8,(255,255,0))
            canvas_point.append(p1)
    
    if len(lb_list) > 0 :
        result_list_x,need_y,p1 = choseFinalFornerDotList(lb_list,'x',shape[0],0,10)
        # #print('************bbbb',lb_list)
        # ##print('************ddddddd',result_list_x)
        if need_y :
            result_list_y,_,p1 = choseFinalFornerDotList(lb_list,'y',shape[0],0,10)
            # ##print('************ddddddd',result_list_y)
            p1 = choseFinalFornerDot(result_list_x,result_list_y)
        # ##print('************ttttt',p1)
        if p1.size == 2 and p1[0]!=np.nan and p1[1]!=np.nan:    
            cv2.circle(img_to_correct,(int(p1[1]),int(p1[0])),8,(255,0,0))
            canvas_point.append(p1)
    if len(rb_list) > 0:
        # #print('************cccccc',rb_list)
        result_list_x,need_y,p1 = choseFinalFornerDotList(rb_list,'x',shape[0],shape[1],10)
        if need_y :
            result_list_y,_,p1 = choseFinalFornerDotList(rb_list,'y',shape[0],shape[1],10)
            p1 = choseFinalFornerDot(result_list_x,result_list_y)
        if p1.size == 2 and p1[0]!=np.nan and p1[1]!=np.nan:
            cv2.circle(img_to_correct,(int(p1[1]),int(p1[0])),8,(0,255,0))
            canvas_point.append(p1)
    if len(ru_list) > 0:
        # #print('************ddddddd',ru_list)
        result_list_x,need_y,p1 = choseFinalFornerDotList(ru_list,'x',0,shape[1],10)
        if need_y :
            result_list_y,_,p1 = choseFinalFornerDotList(ru_list,'y',0,shape[1],10)
            p1 = choseFinalFornerDot(result_list_x,result_list_y)
        if p1.size == 2 and p1[0]!=np.nan and p1[1]!=np.nan:
            cv2.circle(img_to_correct,(int(p1[1]),int(p1[0])),8,(0,0,255))
            canvas_point.append(p1)
    #showImg("canvas_point",img_to_correct)
    # 如果  canvas_point 不等于4 不进行以下的操作
    # #print("canvas_point",canvas_point)  
    if len(canvas_point) == 4:
        #print('solution')
        # #showImg('img_to_correct',img_to_correct)
        width_1 = math.sqrt((canvas_point[0][1]-canvas_point[3][1])*(canvas_point[0][1]-canvas_point[3][1])+(canvas_point[0][0]-canvas_point[3][0])*(canvas_point[0][0]-canvas_point[3][0]))
        width_2 = math.sqrt((canvas_point[1][1]-canvas_point[2][1])*(canvas_point[1][1]-canvas_point[2][1])+(canvas_point[1][0]-canvas_point[2][0])*(canvas_point[1][0]-canvas_point[2][0]))    
        height_1 = math.sqrt((canvas_point[0][0]-canvas_point[1][0])*(canvas_point[0][0]-canvas_point[1][0]) +(canvas_point[0][1]-canvas_point[1][1])*(canvas_point[0][1]-canvas_point[1][1]))
        height_2 = math.sqrt((canvas_point[2][0]-canvas_point[3][0])*(canvas_point[2][0]-canvas_point[3][0]) +(canvas_point[2][1]-canvas_point[3][1])*(canvas_point[2][1]-canvas_point[3][1]))
        
        width = int(min(width_1,width_2) )
        height = int(min(height_1,height_2) )#math.sqrt((canvas_point[0][0]-canvas_point[1][0])*(canvas_point[0][0]-canvas_point[1][0]) +(canvas_point[0][1]-canvas_point[1][1])*(canvas_point[0][1]-canvas_point[1][1]))
        # cv2.imencode('.jpg', img_to_correct)[1].tofile('Archive_to_split_2/harr'+str(flag_of_path)+'.jpg')#img
        # dst = np.array([[0,0],[width ,0],[width,height],[0,height]])
        dst = np.array([[0,0],[height ,0],[height,width],[0,width]])
        # dst = np.array([[0,0],[height ,0],[height,width],[0,width]])
        AffineMatrix1 = cv2.getPerspectiveTransform(np.array(np.float32(canvas_point )),
                                    np.array(np.float32(dst )))
        # ##print(canvas_point)
        # ##print(dst)
        # ##print('ttttttttttAffineMatrix1',AffineMatrix1)
        AffineMatrix, mask = cv2.findHomography(np.float32(canvas_point).reshape(-1,1,2), np.float32(dst).reshape(-1,1,2))#, cv2.RANSAC,5.0)
        # ##print('ttttttttttAffineMatrix',AffineMatrix)
        # AffineImg = cv2.warpPerspective(img_to_correct, M, (width, height))
        
        # Img = img_to_correct # cv2.imread('Archive_to_split_2/harr'+str(flag_of_path)+'.jpg')
        a_begin = int(max(canvas_point[0][0],canvas_point[3][0]))
        a_end  = int(min(canvas_point[1][0],canvas_point[2][0]))

        b_begin = int(max(canvas_point[0][1],canvas_point[1][1]))
        b_end  = int(min(canvas_point[2][1],canvas_point[3][1]))
        # img_to_correct = img_to_correct[a_begin:a_end,b_begin:b_end]
        AffineImg = cv2.warpPerspective(img_to_correct, AffineMatrix,(int( width),int( height)))# (img_to_correct.shape[1], img_to_correct.shape[0]))#

        result_perspective = drawPictureUseCornerDotOri(AffineImg,img_to_correct,AffineMatrix,width,height,canvas_point)
        # result_perspective = reShape(imageModel, result_perspective)

        # ##print('contoursubcontoursubcontoursub',contoursub[0])

        #showImg('AffineImg1',result_perspective)
        # ##print(min_rect)
        # cv2.imencode('.jpg', result_perspective)[1].tofile('Archive_to_split_2/AffineImg'+str(flag_of_path)+'.jpg')#img
        # to return 
        return result_perspective,'perspective'
    # else hough to correct image
    else :
        #print('use hough')
        #showImg('hough',img_to_correct)
        shape_of_hough = img_to_correct.shape
        angle_result = Hough(img_to_correct)
        shape = img_result.shape
        matRotate = cv2.getRotationMatrix2D((shape[0]*0.5,shape[1]*0.5),angle_result,1)    # 为方便旋转后图片能完整展示，所以我们将其缩小
        img_result = cv2.warpAffine(img_result,matRotate,(shape[1],shape[0]))
        shape = img_result.shape
        if abs(angle_result) >= 5.0:
            img_result_1 = cv2.cvtColor(img_result,cv2.COLOR_RGB2GRAY)
            rct, img_result_1 = cv2.threshold(img_result_1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            contoursub, _ = cv2.findContours(img_result_1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            contoursub.sort(key=takeAera, reverse=True)
            if len(contoursub)>0:
                x,y,w,h = cv2.boundingRect(contoursub[0])
                img_result = img_result[y:y+h, x:x+w]
        return img_result,'hough'