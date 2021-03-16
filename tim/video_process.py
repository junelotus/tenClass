# -*- coding: utf-8 -*-
#
#  Media Test
#
#  Created by Li Bin in 2020-07.
#  Copyright (c) 2020 Agora IO. All rights reserved.
#

import os
import pickle

import cv2
import numpy as np

# import common.logger as log
import common.utility as util

PICKLE_MODEL_PATH = util.path_join(util.get_corpus_folder('timestamp'), 'roi_model.pickle')
def showImg(name,img):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name,img)
    cv2.waitKey(1000)

def generate_roi_model(img_path=''):
    """生成model的数据文件，用于ROI查找。
    img_path: model图像的路径，默认值为空，不指定则自动读取corpus/timestamp目录下的roi_model.jpb
    """
    # 打开文件
    if not img_path:
        img_path = util.path_join(util.get_corpus_folder('timestamp'), 'roi_model.jpg')
    img = cv2.imread(img_path)
    # log.logger.dbgout("Read Image: {}", img_path)

    # 二值化
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.normalize(img, img, 0, 1, norm_type=cv2.NORM_MINMAX)

    with open(PICKLE_MODEL_PATH, 'wb') as f:
        pickle.dump(img, f)
        # pickle.dump(selected_contours, f)
    # log.logger.dbgout('Write Pickle File: {}', PICKLE_MODEL_PATH)


def load_roi_model():
    if not os.path.isfile(PICKLE_MODEL_PATH):
        generate_roi_model()

    with open(PICKLE_MODEL_PATH, 'rb') as f:
        img = pickle.load(f)
    return img


def cal_l2_loss(img, img_model):
    y, x = img_model.shape
    img = cv2.resize(img, (x, y), interpolation=cv2.INTER_LINEAR)
    return np.sum(np.square(img - img_model))


def find_roi(img, img_model=None,
             min_width=100, min_height=100,
             l2_loss_threshold=30000,
             ratio=1):
    if img_model is None:
        img_model = load_roi_model()

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 查找Counters并按x排序
    contours, _ = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    list_of_cnt_rect = []
    for cnt in contours:
        x, y, w, h = rect = cv2.boundingRect(cnt)
        if h / w < ratio:
            continue
        if w * h < min_width * min_height or w < min_width or h < min_height:
            continue
        outer_index = next(
            (i for i, c in enumerate(list_of_cnt_rect) if
             x > c[1][0] and y > c[1][1] and x + w <= c[1][0] + c[1][2] and y + h <= c[1][1] + c[1][3]), -1)
        if outer_index >= 0:
            list_of_cnt_rect[outer_index] = (cnt, rect)
        else:
            inner_index = next(
                (i for i, c in enumerate(list_of_cnt_rect) if
                 x < c[1][0] and y < c[1][1] and x + w >= c[1][0] + c[1][2] and y + h >= c[1][1] + c[1][3]), -1)
            if inner_index < 0:
                list_of_cnt_rect.append((cnt, rect))

    cnt_sorted_by_l2loss = []
    for cnt, rect in list_of_cnt_rect:
        (x, y, w, h) = rect
        polygon = cv2.approxPolyDP(cnt, 5, True)[:, 0]
        minimum_rect = cv2.minAreaRect(cnt)
        top_left_x = np.min(polygon[:, 0][np.where(polygon[:, 1] < minimum_rect[0][1])])
        top_left_y = np.min(polygon[:, 1][np.where(polygon[:, 0] == top_left_x)])

        bottom_left_x = np.min(polygon[:, 0][np.where(polygon[:, 1] > minimum_rect[0][1])])
        bottom_left_y = np.max(polygon[:, 1][np.where(polygon[:, 0] == bottom_left_x)])

        top_right_x = np.max(polygon[:, 0][np.where(polygon[:, 1] < minimum_rect[0][1])])
        top_right_y = np.min(polygon[:, 1][np.where(polygon[:, 0] == top_right_x)])

        bottom_right_x = np.max(polygon[:, 0][np.where(polygon[:, 1] > minimum_rect[0][1])])
        bottom_right_y = np.max(polygon[:, 1][np.where(polygon[:, 0] == bottom_right_x)])

        # DEBUG - 用圆圈画出角点
        # cv2.circle(img, (top_left_x, top_left_y), 8, (255, 0, 0))
        # cv2.circle(img, (bottom_left_x, bottom_left_y), 8, (255, 0, 0))
        # cv2.circle(img, (top_right_x, top_right_y), 8, (255, 0, 0))
        # cv2.circle(img, (bottom_right_x, bottom_right_y), 8, (255, 0, 0))

        quad_pts = np.array([[top_left_x - x, top_left_y - y],
                             [bottom_left_x - x, bottom_left_y - y],
                             [top_right_x - x, top_right_y - y],
                             [bottom_right_x - x, bottom_right_y - y]],
                            np.float32)
        print('ddd',quad_pts)
        squre_pts = np.array([[0, 0],
                              [0, 288],
                              [231, 0],
                              [231, 288]],
                             np.float32)
        transform = cv2.getPerspectiveTransform(quad_pts, squre_pts)
        transform1, mask = cv2.findHomography(quad_pts, squre_pts)
        print('transform',transform)
        print('transform1',transform1)
        img_roi = img_gray[y:y + h, x:x + w]
        # showImg('img_roi1',img_roi)
        # showImg('img_gray',img_gray)
        img_roi = cv2.warpPerspective(img_roi, transform, (231, 288))
        # showImg('img_roi',img_roi)
        cv2.normalize(img_roi, img_roi, 0, 1, norm_type=cv2.NORM_MINMAX)
        cnt_sorted_by_l2loss.append((cal_l2_loss(img_roi, img_model), cnt, img_roi))

    cnt_sorted_by_l2loss.sort(key=lambda x: x[0])
    # for item in cnt_sorted_by_l2loss:
    #     log.logger.dbgout('ROI={}, L2_LOSS={}, THRESHOLD={}', cv2.boundingRect(item[1]),
    #                       item[0], l2_loss_threshold)

    return [x[1:] for x in cnt_sorted_by_l2loss]


if __name__ == '__main__':
    import glob

    capture = cv2.VideoCapture('/Users/xhj/Downloads/VID20200814161912.mp4')
    # all_debug_img_paths = glob.glob('/Users/bingo/Downloads/jpg01/*.jpg')
    generate_roi_model()
    path_of_all = glob.glob('PHOTO_IMG_103907_0.jpg')#./normal/*.jpg')
    for p in path_of_all:
        print(p)
        img_frame = cv2.imread(p)
    # while capture.isOpened():
    #     frame_exists, img_frame = capture.read()
        # img_frame = cv2.resize(img_frame, (1040, 780), interpolation=cv2.INTER_LINEAR)
        contours = find_roi(img_frame, img_model=None)
        for n, (cnt, _) in enumerate(contours):
            roi = cv2.boundingRect(cnt)
            cv2.rectangle(img_frame,
                          pt1=(roi[0], roi[1]), pt2=(roi[0] + roi[2], roi[1] + roi[3]),
                          color=(0, 0, 255) if n == 0 else (255, 0, 0),
                          thickness=2)
        for cnt, img in contours:
            cv2.normalize(img, img, 0, 255, norm_type=cv2.NORM_MINMAX)
            # cv2.imshow('ROI', img)
            # cv2.waitKey(500)
            showImg('ROI',img)

cv2.destroyAllWindows()
