# 缩放
import numpy as np
import cv2 as cv
img = cv.imread('messigray.png')
res = cv.resize(img,None,fx=0.2, fy=0.2, interpolation = cv.INTER_CUBIC)
#或者
height, width = img.shape[:2]
res1 = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)
cv.imshow('res', res)
cv.imshow('res1', res1)
k = cv.waitKey(0) & 0xFF    # 64位需有 & 0xFF
if k == 27:         # 等待ESC退出
    cv.destroyAllWindows()  # 破坏所有窗口
elif k == ord('s'): # 等待关键字，保存和退出
    cv.imwrite('messigray2.png',img) # cv.imwrite()保存图像
    cv.destroyAllWindows()

# 平移
# cv.warpAffine函数的第三个参数是输出图像的大小，其形式应为(width，height)。width =列数，height =行数。
import numpy as np
import cv2 as cv
img = cv.imread('messigray.png',0)
rows,cols = img.shape
M = np.float32([[1,0,100],[0,1,50]])
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()

# 旋转
img = cv.imread('messigray.png',0)
rows,cols = img.shape
# cols-1 和 rows-1 是坐标限制
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst = cv.warpAffine(img,M,(cols,rows))
cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()


# 仿射变换
img = cv.imread('chart.png')
rows,cols,ch = img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv.getAffineTransform(pts1,pts2)
dst = cv.warpAffine(img,M,(cols,rows))
from matplotlib import pyplot as plt
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()


# 透视变换
# 图不行
img = cv.imread('chart2.png')
rows,cols,ch = img.shape
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(300,300))
# from matplotlib import pyplot as plt
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()