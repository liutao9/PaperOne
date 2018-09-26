import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from MethodOur import function
import time
# 圆环
# ct, ori, spac = function.load_itk("../Data/test2.mhd")
# z_coordinate = np.loadtxt('../MethodOur/result/z1.txt')
# f = open('result/result1.txt', 'w')
# 带有狭窄的圆环
# ct, ori, spac = function.load_itk("../Data/test3.mhd")
# z_coordinate = np.loadtxt('../MethodOur/result/z2.txt')
# f = open('result/result2.txt', 'w')
# 真实血管
ct, ori, spac = function.load_itk("../Data/Confidence_Connected_Yi_Bao_Jun_4.mhd")
z_coordinate = np.loadtxt('../MethodOur/result/z3.txt')
f = open('result/result3.txt', 'w')

fina_x = []
S = []
count = 0
time_start = time.time()
for i in z_coordinate:
    img = ct[int(i), :, :]
    #img = img[0:128, :]
    img = img[190:240, 100:170]           ##真实血管中截取部分
    img = np.uint8(img)
    imgray = cv2.Canny(img, 600, 100, 3)#Canny边缘检测，参数可更改  
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#contours为轮廓集，可以计算轮廓的长度、面积等  
    cnt = contours[0]
    S1 = cv2.contourArea(cnt)
    ell = cv2.fitEllipse(cnt)
    S2 = math.pi*(ell[1][0]/2)*(ell[1][1]/2)
    img = cv2.ellipse(img, ell, (0, 255, 0), 2)
    print(str(i)+"    "+str(S2)+"   "+str(ell[0][0])+"   "+str(ell[0][1]))
    f.write('{0}\n'.format(S2))
    count = count + 1
    fina_x.append(count)
    S.append(S2)

f.close()
time_end = time.time()
print('cost time is :{0},mean of the cost time is :{1} '.format(time_end-time_start, (time_end-time_start)/len(z_coordinate)))

plt.plot(fina_x, S)
plt.xlim(0.0, fina_x[len(fina_x) - 1])
plt.ylim(20.0, max(S)+20)
plt.show()
