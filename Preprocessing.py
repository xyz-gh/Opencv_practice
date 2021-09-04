import cv2
import numpy as np
from PIL import Image
 
'''水平投影'''
def getHProjection(image):
    hProjection = np.zeros(image.shape,np.uint8)
    #图像高与宽
    (h,w)=image.shape
    #长度与图像高度一致的数组
    h_ = [0]*h
    #循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y,x] == 255:
                h_[y]+=1
    #绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y,x] = 255
    cv2.imshow('hProjection2',hProjection)
    cv2.imwrite(r"C:/Users/86155/Desktop/2.jpg",hProjection)
 
    return h_
 
if __name__ == "__main__":
    #读入原始图像
    origineImage = cv2.imread(r"C:\\Users\\86155\\Desktop\\005052.jpg")
    # 图像灰度化   
    #image = cv2.imread('test.jpg',0)
    image = cv2.cvtColor(origineImage,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',image)
    # 将图片二值化
    retval, img = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow('binary',img)
    #图像高与宽
    (h,w)=img.shape
    Position = []
    #水平投影
    H = getHProjection(img)
 
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