import function
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import math
'''
#圆环面积测试
start_point = '64 108 110'
end_point = '66 102 144'
ct, ori, spac = function.load_itk("../Data/test2.mhd")
ct2, ori2, spac2 = function.load_itk("../Data/test2C.mhd")
f = open('result/result1.txt', 'w')
f_z = open('result/z1.txt', 'w')
'''

'''
#带有狭窄的圆环的测试
start_point = '64 108 110'
end_point = '66 102 144'
ct, ori, spac = function.load_itk("../Data/test3.mhd")
ct2, ori2, spac2 = function.load_itk("../Data/test2C.mhd")
f = open('result/result2.txt', 'w')
f_z = open('result/z2.txt', 'w')
'''

#'''
#真实的血管
start_point = '123 204 98'
end_point = '141 218 82'
ct, ori, spac = function.load_itk("../Data/Confidence_Connected_Yi_Bao_Jun_4.mhd")
ct2, ori2, spac2 = function.load_itk("../Data/Confidence_Connected_Yi_Bao_Jun_4C.mhd")
f = open('result/result3.txt', 'w')
f_z = open('result/z3.txt', 'w')
#'''

x_space = spac[1]
y_space = spac[2]
scale1 = np.array([[spac[0], 0.0, 0.0], [0.0, spac[1], 0.0], [0.0, 0.0, spac[2]]])             ###scale1 就是Λ = diag (α, β, γ) 对角矩阵
x_scale = spac[1] / 4
y_scale = spac[2] / 4
scale2 = np.array([[x_scale, 0.0, 0.0], [0.0, y_scale, 0.0], [0.0, 0.0, 1.0]])                 ###scale2 就是Λ′ = diag(α′, β′, 1) 平面缩放比例
width = 60                                                                                     ### 生成图像长宽，可调为50
height = 60
step = spac[1] * 1.5                                                                             ###参数 步长 可调为 1.5
begin = 0
fina_x = []
S = []
start_point = start_point.split(' ')
start_point.reverse()
end_point = end_point.split(' ')
end_point.reverse()
for i in range(3):
    start_point[i] = int(start_point[i])
    end_point[i] = int(end_point[i])
oldnorm = np.array([end_point]) - np.array([start_point])
current_i = 0
time_start = time.time()
while function.stop(np.array(start_point), end_point, 5):
    x = []
    y = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if ct2[start_point[0] + i][start_point[1] + j][start_point[2] + k] == 255:
                    x.append([start_point[0] + i, start_point[1] + j, start_point[2] + k])
    sample = np.array(x)
    sample = sample.dot(scale1)
    meanVal = np.mean(sample, axis=0)
    covMat = np.cov(sample - meanVal, rowvar=0)
    eigVal, eigVects = np.linalg.eig(np.mat(covMat))
    maxvec = eigVects[:, np.argsort(eigVal)[-1]].T
    if oldnorm.dot(maxvec.T) < 0:
        maxvec = -maxvec
    normalMat = function.calcNorm(maxvec.tolist()[0])
    x_trans = np.linalg.inv(scale1).dot(meanVal) - np.linalg.inv(scale1).dot(normalMat.dot(scale2.dot(np.array([width / 2.0, height / 2.0, 0.0]))))
    x_trans = x_trans.tolist()
    x_trans.append(1.0)
    transMat = np.zeros([4, 4])
    transMat[0:3, 0:3] += np.linalg.inv(scale1).dot(normalMat)
    transMat[:, 3] += x_trans
    image = np.zeros([width, height])
    for i in range(width):
        for j in range(height):
            pos = transMat.dot(np.array([x_scale * i, y_scale * j, 0.0, 1.0]))
            tmp = [0, 0, 0]
            ans = int(round(function.calcScale(pos.tolist(), ct)))
            image[i][j] = ans
            for k in range(3):
                tmp[k] = int(round(pos[k]))
            pos = tmp
            if ct[pos[0]][pos[1]][pos[2]] == 255:
                y.append(pos)
    image = cv2.GaussianBlur(image, (5, 5), 1.5)
    fina_x.append(begin)
    S.append(np.sum(image) / 255 * x_scale * y_scale)
    begin += step
    meanVal = np.mean(np.array(y), axis=0)
    print('{0} {1} {2}'.format(start_point, np.sum(image) / 255 * x_scale * y_scale, current_i))
    #f.write('{0}\n'.format(np.sum(image) / 255 * x_scale * y_scale))
    f_z.write('{0}\n'.format(start_point[0]))
    start_point = (meanVal + step * np.linalg.inv(scale1).dot(maxvec.T).T).tolist()[0]
    oldNorm = maxvec.copy()
    for i in range(3):
        start_point[i] = int(round(start_point[i]))
    if current_i >= 100:
        print('fail')
        break
    current_i += 1
time_end = time.time()
print('cost time is :{0},mean of the cost time is :{1} '.format(time_end-time_start, (time_end-time_start)/current_i))
print('fina_x :{0}'.format(fina_x))
print('S: {0}'.format(S))
plt.plot(fina_x, S)
plt.xlim(0.0, fina_x[len(fina_x) - 1])
plt.ylim(0.0, max(S)+1)
S = np.array([S]).T

mean = np.mean(S, axis=0)
print('the mean is :{0}'.format(mean))
cov = np.cov(S-mean, rowvar=0)
print('the cov is: {0} ,the math.sqrt(cov) is: {1} , the math.sqrt(cov)/mean is {2} '.format(cov, math.sqrt(cov), math.sqrt(cov)/mean))
f.close()
f_z.close()
plt.show()
