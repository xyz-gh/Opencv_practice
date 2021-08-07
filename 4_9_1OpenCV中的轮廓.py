import numpy as np
import cv2 as cv
im = cv.imread('chart.png')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
# 查找轮廓
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(hierarchy)
# 绘制轮廓
# 绘制所有轮廓
cv.drawContours(im, contours, -1, (0,255,0), 3)
# 绘制单个轮廓
cv.drawContours(im, contours, 3, (0,255,0), 3) 
# 3rd way
cnt = contours[4]
cv.drawContours(im, [cnt], 0, (0,255,0), 3)

cv.imshow('img',im)
cv.waitKey(0)
cv.destroyAllWindows()