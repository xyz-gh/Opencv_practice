# 颜色转换的标记
import cv2 as cv
flags = [i for i in dir(cv) if i.startswith('COLOR_')]
print( flags )

# 对象追踪：蓝色
import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
while(1):
    # 读取帧
    _, frame = cap.read()
    # 转换颜色空间 BGR 到 HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # 定义HSV中蓝色的范围
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # 设置HSV的阈值使得只取蓝色
    # 高值和低值变为0，中间值255
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # 将掩膜和图像逐像素相加
    # 白色（255，255，255） 黑色（0，0，0）
    # 白色区域保留，黑色区域剔除
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()

green = np.uint8([[[0,255,0 ]]])
hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
print( hsv_green )

# 同时提取b g r
import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)

while(1):
    # 读取帧
    _, frame = cap.read()
    # 转换颜色空间 BGR 到 HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # 定义范围（不知如何是好）
    # 定义HSV中蓝色的范围
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # 定义HSV中绿色的范围
    lower_green = np.array([100,50,50])
    upper_green = np.array([130,255,255])
    # 定义HSV中红色的范围
    lower_red = np.array([0,50,50])
    upper_red = np.array([20,255,255])

    # 设置HSV的阈值使得只取蓝色
    mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
    # 设置HSV的阈值使得只取绿色
    mask_green = cv.inRange(hsv, lower_green, upper_red)
    # 设置HSV的阈值使得只取红色
    mask_red = cv.inRange(hsv, lower_red, upper_red)

    # 将掩膜和图像逐像素相加
    res_blue = cv.bitwise_and(frame,frame, mask= mask_blue)
    res_green = cv.bitwise_and(frame,frame, mask= mask_green)
    res_red = cv.bitwise_and(frame,frame, mask= mask_red)
    res_bg = cv.add(res_blue, res_green)
    res = cv.add(res_bg, res_red)
    
    cv.imshow('frame',frame)
    cv.imshow('mask_blue',mask_blue)
    cv.imshow('mask_green',mask_green)
    cv.imshow('mask_red',mask_red)
    cv.imshow('res_bg',res_bg)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()