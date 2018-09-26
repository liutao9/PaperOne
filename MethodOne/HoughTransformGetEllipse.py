import numpy as np
import math
from skimage import draw, transform, feature
from MethodOur import function
import matplotlib.pyplot as plt
import time
# 圆环测试
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
    img = img[190:240, 100:170]
    result = transform.hough_ellipse(img)
    result.sort(order='accumulator') #根据累加器排序

    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]
    area = math.pi * a * b
    print('{0}: {1}'.format(i, area))
    f.write('{0}\n'.format(area))
    count = count + 1
    fina_x.append(count)
    S.append(area)

f.close()
time_end = time.time()
print('cost time is :{0},mean of the cost time is :{1} '.format(time_end-time_start, (time_end-time_start)/len(z_coordinate)))

plt.plot(fina_x, S)
plt.xlim(0.0, fina_x[len(fina_x) - 1])
plt.ylim(20.0, max(S)+20)
plt.show()
