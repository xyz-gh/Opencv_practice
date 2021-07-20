# 绘图
import numpy as np
import cv2 as cv
# 厚度：线或圆等的粗细。如果对闭合图形（如圆）传递-1 ，它将填充形状。默认厚度= 1
# 创建黑色的图像
img = np.zeros((512,512,3), np.uint8)
# 绘制一条厚度为5的蓝色对角线
cv.line(img,(0,0),(511,511),(255,0,0),5)
# 右上角绿色矩形
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
# 矩形内的圆
cv.circle(img,(447,63), 63, (0,0,255), -1)
# 椭圆
# cv2.ellipse(image, centerCoordinates, axesLength, angle, startAngle, endAngle, color [, thickness[, lineType[, shift]]])
cv.ellipse(img,(256,256),(100,50),0,0,360,255,-1)
# 多边形
# polylines(img, pts, isClosed, color[, thickness[, lineType[, shift]]])
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(0,255,255))
# 向图像添加文本
# cv2.putText(img, str(i), (123,456)), font, fontsize, (0,255,0), 字体粗细)
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)

cv.imshow('image',img)
k = cv.waitKey(0) & 0xFF    # 64位需有 & 0xFF
if k == 27:         # 等待ESC退出
    cv.destroyAllWindows()  # 破坏所有窗口
elif k == ord('s'): # 等待关键字，保存和退出
    cv.imwrite('messigray.png',img) # cv.imwrite()保存图像
    cv.destroyAllWindows()