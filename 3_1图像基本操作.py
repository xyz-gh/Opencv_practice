import numpy as np
import cv2 as cv
img = cv.imread('messigray.png')

# 访问和修改像素值
#blue = img[100,100,0]
#print( blue )
img[100,100] = [255,255,255]
print( img[100,100] )
px = img[100,100]
print(px)
blue = img[100,100,0]
print(blue)

# 另一种访问和修改方法
# 不能用
img.item(100,100,0)
img.itemset((10,10,2),100)
img.item(100,100,0)

# 访问图像属性
# 形状
# 
print(img.shape)    # 高、宽、通道
# 像素总数
print(img.size)
# 图像数据类型
print(img.dtype)

# 图像感兴趣区域ROI
ball = img[0:200, 0:200]
img[880:1080, 1720:1920] = ball
cv.imshow('image',img)
k = cv.waitKey(0) & 0xFF    
if k == 27:         # 等待ESC退出
    cv.destroyAllWindows()  # 破坏所有窗口
elif k == ord('s'): # 等待关键字，保存和退出
    cv.imwrite('messigray2.png',img) # cv.imwrite()保存图像
    cv.destroyAllWindows()

# b g r
# 拆分图像通道
b,g,r = cv.split(img)   # 耗时，一般用numpy索引
# 合并图像通道
img = cv.merge((b,g,r))
# b通道第一层
b = img [:, :, 0]
# 将红色设置为0
img [:, :, 2] = 0

# void copyMakeBorder( InputArray src,  OutputArray dst,int top,  int bottom,  int left,  int right,  int borderType,const Scalar& value = Scalar())
# https://blog.csdn.net/sss_369/article/details/92759383
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
BLUE = [255,0,0]
img1 = cv.imread('opencvlogo.png')
replicate = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REPLICATE)
reflect = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT)
reflect101 = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_REFLECT_101)
wrap = cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_WRAP)
constant= cv.copyMakeBorder(img1,10,10,10,10,cv.BORDER_CONSTANT,value=BLUE)
plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()