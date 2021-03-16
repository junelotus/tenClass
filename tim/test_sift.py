import numpy as np
import cv2
from matplotlib import pyplot as plt

from matplotlib import pyplot as plt
import cv2
import numpy as np
def showImg(name,img):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name,img)
    cv2.waitKey(10000)
img = cv2.imread('jpg15/dabaozhi.jpg')
rows,cols = img.shape[:2]
# showImg('dabaozhi',img)
pts1 = np.float32([[368,514],[554,509],[560,736],[362,711]])
pts2 = np.float32([[368,514],[368+200,514],[368+200,514+200],[368,514+200]])
AffineMatrix = cv2.getPerspectiveTransform(np.array(np.float32(pts1 )),
                                    np.array(np.float32( pts2)))
    # Img = img_to_harr # cv2.imread('Archive_to_split_2/harr'+str(flag_of_path)+'.jpg')
AffineImg = cv2.warpPerspective(img, AffineMatrix, (int(img.shape[1]), int(img.shape[0])))
showImg('AffineImg',AffineImg)
# M = cv2.getAffineTransform(pts1,pts2)
# #第三个参数：变换后的图像大小
# res = cv2.warpAffine(img,M,(rows,cols))
# # cv2.imencode('.jpg', res)[1].tofile('hanjune.jpg')#img        
# plt.subplot(121)
# plt.imshow(img[:,:,::-1])

# plt.subplot(122)
# plt.imshow(res[:,:,::-1])

# plt.show()


# a = [1,2,3,4,5,6,7,8,9,10]
# print(a)
# print(a[2:9])
# print(a[10:])
