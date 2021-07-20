# numpy https://blog.csdn.net/a373595475/article/details/79580734
# 使用轨迹栏创建颜色和画笔半径可调的Paint应用程序

import numpy as np
import cv2 as cv
def nothing(x):
    pass

# 原代码 img = np.zeros((512,512,3), np.uint8)
# 创建一个白色的图像，一个窗口，并绑定到窗口的功能
img = np.full((512,512,3), 255, np.uint8)   
cv.namedWindow('image')
# cv.imshow('image', img)

drawing = False # 如果按下鼠标，则为真
# ix,iy = -1,-1
# 鼠标回调函数
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
                cv.circle(img,(x,y),thickness,color,-1) 
                # 原代码 cv.circle(img,(x,y),5,(0,0,255),-1)
                # 修改 粗细、颜色 为可调变量
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.circle(img,(x,y),thickness,color,-1)  

# 创建颜色变化的轨迹栏
cv.createTrackbar('R','image',0,255,nothing)
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)
cv.createTrackbar('Thickness','image',1,10,nothing)
# 为 ON/OFF 功能创建开关
#switch = '0 : OFF \n1 : ON'
#cv.createTrackbar(switch, 'image',0,1,nothing)

color=(0,0,0)
thickness=1
cv.setMouseCallback('image',draw_circle,(0,0,0))
while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    # 得到四条轨迹的当前位置
    r = cv.getTrackbarPos('R','image')
    g = cv.getTrackbarPos('G','image')
    b = cv.getTrackbarPos('B','image')
    thickness = cv.getTrackbarPos('Thickness','image')
    color = [b,g,r]
cv.destroyAllWindows()