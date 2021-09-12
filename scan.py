import cv2
import numpy as np


## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None,
                                            scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y],
                                                  cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d][c]) * 13 + 27,
                               30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d][c],
                            (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver


def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def drawRectangle(img, biggest, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]),
             (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]),
             (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]),
             (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]),
             (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img


def nothing(x):
    pass


def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)


def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1, Threshold2
    return src


import cv2
import numpy as np
from skimage.filters import threshold_local
# import utlis

import matplotlib.pyplot as plt
# matplotlib inline


# 需要读取的图片
pathImage = "C:/Document-Scanner-master/1.jpg"
# pathImage = "dmbj2.jpeg"
# pathImage = "page3.jpg"


from IPython.display import Image
Image(filename = pathImage, width=500)

count=0

# 读取图片
img = cv2.imread(pathImage, 3)
# 解决偏色问题
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])

print('img')
plt.figure(figsize=(16, 8))
plt.imshow(img)
plt.show()



# img = cv2.imread(pathImage)
img = cv2.imread(pathImage, 3)
# 解决matplotlib偏色问题
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])

orig = img.copy()
print(img.shape)

# 设置图像大小,A4纸的大小
heightImg = 3508
widthImg = 2479
# img = imutils.resize(img, width=widthImg, height = heightImg) # 500
img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE

# 设置图像大小
# heightImg = int(img.shape[0])
# widthImg = int(img.shape[1])



imgBlank = np.zeros(
    (heightImg, widthImg, 3),
    np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR

# 滑动条，不需要了
# thres = valTrackbars() # utlis.valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
# imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # APPLY CANNY BLUR

imgThreshold = cv2.Canny(imgBlur, 200, 200) # 更改这个200*200来调整框

kernel = np.ones((5, 5))
imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION



# 看下绿色框效果
plt.figure(figsize=(16, 16))
ax1 = plt.subplot(2, 2, 1, frameon = False) # 两行一列，位置是1的子图
plt.title('img')
plt.imshow(img)

ax2 = plt.subplot(2, 2, 2, frameon = False) # 两行一列
plt.title('imgGray')
plt.imshow(imgGray)

ax3 = plt.subplot(2, 2, 3, frameon = False) # 两行一列
plt.title('imgThreshold')
plt.imshow(imgThreshold)

plt.show()


# plt.figure(figsize=(16, 8))
# plt.title('img')
# plt.imshow(img)
# plt.show()


# plt.figure(figsize=(10, 6))
# plt.title('imgGray')
# plt.imshow(imgGray)
# plt.show()



# plt.figure(figsize=(10, 6))
# plt.title('imgThreshold')
# plt.imshow(imgThreshold)
# plt.show()



## 找边界
## FIND ALL COUNTOURS
imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(
    imgThreshold, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS

plt.figure(figsize=(10, 6))
plt.title('imgContours')
plt.imshow(imgContours)
plt.show()

# 寻找最大边界，对于背景色区分不明显的，后面增加手动框选
# FIND THE BIGGEST COUNTOUR
biggest, maxArea = biggestContour(contours) # utlis.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
if biggest.size != 0:
    biggest = reorder(biggest) # utlis.reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0),
                     20)  # DRAW THE BIGGEST CONTOUR
    imgBigContour = drawRectangle(imgBigContour, biggest, 2) # utlis.drawRectangle(imgBigContour, biggest, 2)
    pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg],
                       [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgOri = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    #REMOVE 20 PIXELS FORM EACH SIDE
    imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20,
                                    20:imgWarpColored.shape[1] - 20]
    imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

    # APPLY ADAPTIVE THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
    imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
    imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

    # Image Array for Display
    imageArray = ([img, imgGray, imgThreshold, imgContours], [
        imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre
    ])

else:
    imageArray = ([img, imgGray, imgThreshold,
                   imgContours], [imgBlank, imgBlank, imgBlank, imgBlank])

print(biggest)

# [[[ 154  208]]

# [[2261  160]]

# [[ 168 3356]]

# [[2301 3387]]]


# 显示全部的图
# LABELS FOR DISPLAY
# lables = [["Original","Gray","Threshold","Contours"],
#           ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]

# stackedImage = utlis.stackImages(imageArray,0.75,lables)
# plt.figure(figsize=(10, 6))
# plt.title('imgContours')
# plt.imshow(stackedImage)
# plt.show()


# imageArray = ([img, imgGray, imgThreshold, imgContours], [
#         imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre
#     ])

plt.figure(figsize=(16, 16))
ax1 = plt.subplot(2, 4, 1, frameon = False) 
plt.title('img')
plt.imshow(img)

ax2 = plt.subplot(2, 4, 2, frameon = False)
plt.title('imgGray')
plt.imshow(imgGray)

ax3 = plt.subplot(2, 4, 3, frameon = False) 
plt.title('imgThreshold')
plt.imshow(imgThreshold)

ax3 = plt.subplot(2, 4, 4, frameon = False)
plt.title('imgContours')
plt.imshow(imgContours)

ax3 = plt.subplot(2, 4, 5, frameon = False)
plt.title('imgBigContour')
plt.imshow(imgBigContour)

ax3 = plt.subplot(2, 4, 6, frameon = False)
plt.title('imgWarpColored')
plt.imshow(imgWarpColored)

ax3 = plt.subplot(2, 4, 7, frameon = False)
plt.title('imgWarpGray')
plt.imshow(imgWarpGray)

ax3 = plt.subplot(2, 4, 8, frameon = False)
plt.title('imgAdaptiveThre')
plt.imshow(imgAdaptiveThre)

plt.show()



# 保存图片，不清晰
# cv2.imwrite("Scanned/myImage"+str(count)+".jpg", imgWarpColored)


# 二值化，用高斯gaussian

# imageArray = ([img,imgGray,imgThreshold,imgContours],
#                   [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre])


warped = imgWarpGray

# # from skimage.filters import threshold_local
T = threshold_local(warped, 81, offset = 11, method = "gaussian")  # 101， 11
warped = (warped > T).astype("uint8") * 255
print(warped)
# [[255 255 255 ... 255 255 255]
# [255 255 255 ... 255 255 255]
# [255 255 255 ... 255 255 255]
#  ...
# [255 255 255 ... 255 255 255]
# [255 255 255 ... 255 255 255]
# [255 255 255 ... 255 255 255]]


from PIL import Image
im = Image.fromarray(warped)
im.save(f"Scanned/new_{pathImage}")

from IPython.display import Image
from IPython.core.display import HTML, display_jpeg 

PATH = f"Scanned/new_{pathImage}"
Image(filename = PATH, width=500)



# 裁剪后的效果
from PIL import Image
im = Image.fromarray(imgWarpColored)
im.save(f"Scanned/new_{pathImage}")

from IPython.display import Image
from IPython.core.display import HTML 

PATH = f"Scanned/new_{pathImage}"
Image(filename = PATH, width=500)

from PIL import Image
im = Image.fromarray(img)
im.save(f"Scanned/new_img_{pathImage}")
PATH1 = f"Scanned/new_img_{pathImage}"


im = Image.fromarray(warped)
im.save(f"Scanned/new_imgwarped_{pathImage}")
PATH2 = f"Scanned/new_imgwarped_{pathImage}"


display_jpeg(HTML(f"<table><tr><td><img src='{PATH1}'></td><td><img src='{PATH2}'></td></tr></table>"))
