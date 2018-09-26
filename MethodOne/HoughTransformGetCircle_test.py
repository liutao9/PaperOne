import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from skimage import draw, transform, feature
from MethodOur import function

ct, ori, spac = function.load_itk("../Data/test3.mhd")
img = ct[110, :, :]
img = img[0:128, :]
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fig, (ax0, ax1) = plt.subplots(1, 2)

ax0.imshow(img)
ax0.set_title('origin image')
hough_radii = np.arange(1, 10, 0.1)                          #半径范围
hough_res = transform.hough_circle(img[:, :], hough_radii) #圆变换 返回值是检测到的多个符合条件的圆环图像，[nums, x, y]
centers = []                                               #保存所有圆心点坐标
accums = []                                                #累积值
radii = []                                                 #半径

for radius, h in zip(hough_radii, hough_res):
 num_peaks = 1                                             #每一个半径值，选取一个圆
 peaks = feature.peak_local_max(h, num_peaks=num_peaks)    #取出峰值，坐标，其实就是圆心坐标
 centers.extend(peaks)
 accums.extend(h[peaks[:, 0], peaks[:, 1]])
 radii.extend([radius] * num_peaks)                        #[]是因为类型不匹配

image = np.copy(img)
for idx in np.argsort(accums)[::-1][:1]:                   #[:num]控制圆环个数,找到最合适的圆
 center_x, center_y = centers[idx]
 radius = radii[idx]
 print(radius)
 print(radius * radius * math.pi)
 cx, cy = draw.circle_perimeter(int(center_y), int(center_x), int(radius))
 image[cy, cx] =(125)

ax1.imshow(image)
ax1.set_title('detected image')
plt.show()
