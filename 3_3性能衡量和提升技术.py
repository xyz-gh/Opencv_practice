import numpy as np
import cv2 as cv

# cv.getTickCount 函数返回从参考事件到调用此函数那一刻之间的时钟周期数。
# cv.getTickFrequency函数返回时钟周期的频率或每秒的时钟周期数。
img1 = cv.imread('messigray.png')
e1 = cv.getTickCount()
for i in range(5,49,2):
    img1 = cv.medianBlur(img1,i)
e2 = cv.getTickCount()
t = (e2 - e1)/cv.getTickFrequency()
print( t )
# 我得到的结果是3.1211838秒

# 检查是否启用了优化
'''
In [5]: cv.useOptimized()
Out[5]: True
In [6]: %timeit res = cv.medianBlur(img,49)
10 loops, best of 3: 34.9 ms per loop
# 关闭它
In [7]: cv.setUseOptimized(False)
In [8]: cv.useOptimized()
Out[8]: False
In [9]: %timeit res = cv.medianBlur(img,49)
10 loops, best of 3: 64.1 ms per loop

# 以 x 的平方为例测试
In [10]: x = 5
In [11]: %测时 y=x**2
10000000 loops, best of 3: 73 ns per loop
In [12]: %测时 y=x*x
10000000 loops, best of 3: 58.3 ns per loop
In [15]: z = np.uint8([5])
In [17]: %测时 y=z*z
1000000 loops, best of 3: 1.25 us per loop
In [19]: %测时 y=np.square(z)
1000000 loops, best of 3: 1.16 us per loop

# 比较 cv.countNonZero 和 np.count_nonzero
In [35]: %测时 z = cv.countNonZero(img) 
100000 loops, best of 3: 15.8 us per loop
In [36]: %测时 z = np.count_nonzero(img) 
1000 loops, best of 3: 370 us per loop
'''

# OpenCV函数比Numpy函数要快
# 尽量避免在Python中使用循环，尤其是双/三重循环等
# 由于Numpy和OpenCV已针对向量运算进行了优化，因此将算法/代码向量化到最大程度
# 利用缓存一致性
# 除非需要，否则切勿创建数组的副本。尝试改用视图