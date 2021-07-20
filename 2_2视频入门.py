# 从相机中读取视频
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # 我们在框架上的操作到这里
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv::cvtColor()支持多种颜色空间之间的转换  
    # 详解 https://blog.csdn.net/guduruyu/article/details/68941554
    # 显示结果帧e
    cv.imshow('frame', gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# 完成所有操作后，释放捕获器
cap.release()
cv.destroyAllWindows()


# 从文件播放视频
import numpy as np
import cv2 as cv
cap = cv.VideoCapture('video.ts')
while cap.isOpened():
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
# 显示框架时使用适当的时间cv.waitKey()。
# 如果太小，则视频将非常快，而如果太大，则视频将变得很慢。正常情况下25毫秒。


# 保存视频
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
# 定义编解码器并创建VideoWriter对象
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480), True)
# VideoWriter(filename, fourcc, fps, frameSize[, isColor]) -> <VideoWriter object>
# 详解 https://blog.csdn.net/weixin_36670529/article/details/100977537
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame, 0)
    # 写翻转的框架
    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# 完成工作后释放所有内容
cap.release()
out.release()
cv.destroyAllWindows()