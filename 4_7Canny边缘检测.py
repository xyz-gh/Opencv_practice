import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# edges	=	cv.Canny(	image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]	)
# edges	=	cv.Canny(	dx, dy, threshold1, threshold2[, edges[, L2gradient]]	)

img = cv.imread('messigray.png',0)
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# 编写一个小应用程序以找到Canny边缘检测，该检测的阈值可以使用两个跟踪栏进行更改。
import numpy as np
import cv2 as cv
def p(x):
    edges = cv.Canny(img,minval,maxval)
    cv.imshow('edges', edges)
# 创建一个黑色的图像，一个窗口
img1 = np.full((300,512,3), 255, np.uint8)
cv.namedWindow('image1')

# 创建颜色变化的轨迹栏
cv.createTrackbar('minval','image1',0,255,p)
cv.createTrackbar('maxval','image1',0,255,p)
minval=1
maxval=1

img = cv.imread('messigray.png',0)
cv.imshow('image', img)

while(1):
    cv.imshow('image1',img1)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # 得到轨迹的当前位置
    minval = cv.getTrackbarPos('minval','image1')
    maxval = cv.getTrackbarPos('maxval','image1')

cv.destroyAllWindows()