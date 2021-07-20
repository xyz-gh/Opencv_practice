import numpy as np
import cv2 as cv
img = cv.imread('pic.jpg',0)  
# 1 cv.IMREAD_COLOR： 加载彩色图像。任何图像的透明度都会被忽视。它是默认标志。
# 0 cv.IMREAD_GRAYSCALE：以灰度模式加载图像
# -1 cv.IMREAD_UNCHANGED：加载图像，包括alpha通道
cv.imshow('image',img)
k = cv.waitKey(0) & 0xFF    # 64位需有 & 0xFF
# cv.waitKey()是一个键盘绑定函数。其参数是以毫秒为单位的时间。
# 该函数等待任何键盘事件指定的毫秒。如果您在这段时间内按下任何键，程序将继续运行。
# 如果**0**被传递，它将无限期地等待一次敲击键。它也可以设置为检测特定的按键
if k == 27:         # 等待ESC退出
    cv.destroyAllWindows()  # 破坏所有窗口
elif k == ord('s'): # 等待关键字，保存和退出
    cv.imwrite('messigray.png',img) # cv.imwrite()保存图像
    cv.destroyAllWindows()  # 破坏特定窗口 cv.destroyWindow()

# matplotlib显示图像
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('pic.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # 隐藏 x 轴和 y 轴上的刻度值
plt.show()