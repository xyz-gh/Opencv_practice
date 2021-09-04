from PIL import Image
from imutils import perspective
from skimage.filters import threshold_local
import cv2
import numpy as np
import imutils
 


# 水平投影
def getHProjection(image):
    hProjection = np.zeros(image.shape,np.uint8)
    # 图像高与宽
    (h,w)=image.shape
    # 长度与图像高度一致的数组
    h_ = [0]*h
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y,x] == 255:
                h_[y]+=1
    # 绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y,x] = 255
    cv2.imshow('hProjection2',hProjection)
    cv2.imwrite(r"C:/Users/86155/Desktop/2.jpg",hProjection)
 
    return h_
 


# 图像预处理
if __name__ == "__main__":
    # 读入原始图像
    origineImage = cv2.imread(r"C:\\Users\\86155\\Desktop\\005052.jpg")
    # 比例
    orig = origineImage.copy()
    # 图像灰度化   
    image = cv2.cvtColor(origineImage,cv2.COLOR_BGR2GRAY)
    # 滤波
    gray_image = cv2.GaussianBlur(image, (5, 5), 0)
    # 将图片二值化
    retval, gray_img = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY_INV)
    # 边缘检测
    canny_image = cv2.Canny(gray_img, 70, 200)
    

    # 梯形校正
    # 查找轮廓
    contours = cv2.findContours(canny_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 保留最大轮廓
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018*peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        print("No contour detected")
    else:
        # 视角
        warped = perspective.four_point_transform(orig, screenCnt.reshape(4, 2))
        # 灰度转换
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # 阈值分割
        T = threshold_local(warped, 11, offset=10, method='gaussian')
        warped = (warped > T).astype('uint8') * 255
        # cv2.imshow('keystone_correction.png',warped)
        # 校正后获得的图片名为 warped
        

    #图像高与宽
    (h,w)=warped.shape
    Position = []
    #水平投影
    H = getHProjection(warped)
 
    start = 0
    H_Start = []
    H_End = []
    temp = []
    #根据水平投影获取垂直分割位置
    for i in range(len(H)):
        print(H[i])
        if H[i] > 0 and start == 0:
            H_Start.append(i)
            start = 1
        if H[i] <= 0 and start == 1:
            H_End.append(i)
            start = 0
    print(H_Start)
    print(H_End)
    #分割行，分割之后再进行列分割并保存分割位置
    for i in range(len(H_End)):
        #获取行图像
        temp.append((H_End[i]-H_Start[i]))
    temp = sorted(temp)
    print(temp)
    balence =temp[int(len(temp)/2)]
    print(balence)
    for i in range(len(H_End)):
        if(2*(H_End[i]-H_Start[i])>balence):
            print(i)
            print(H_End[i] - H_Start[i])
            cropImg = origineImage[H_Start[i]:H_End[i], 0:w]
            cv2.imwrite("C:/Users/86155/Desktop/image/"+str(i).zfill(3)+".jpg", cropImg)