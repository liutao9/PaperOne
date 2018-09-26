import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
img = np.loadtxt('../MethodTwo/result/result2.txt')
img = np.uint8(img)
ax0.imshow(img)
imgray = cv2.Canny(img, 600, 100, 3)#Canny边缘检测，参数可更改  
ax1.imshow(imgray)
ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#contours为轮廓集，可以计算轮廓的长度、面积等  
for cnt in contours:
  if len(cnt) > 1:
     S1 = cv2.contourArea(cnt)
     ell = cv2.fitEllipse(cnt)
     S2 = math.pi*ell[1][0]*ell[1][1]
     img = cv2.ellipse(img, ell, (0, 255, 0), 2)
     print(str(S1)+"    "+str(S2)+"   "+str(ell[0][0])+"   "+str(ell[0][1]))
     ax2.imshow(img)

plt.show()
