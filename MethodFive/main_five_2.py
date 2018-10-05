import MethodOur.function
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import time

def main():

    #带有狭窄的圆环的测试
    start_point = '64 105 110'
    end_point = '66 102 144'
    ct, ori, spac = MethodOur.function.load_itk("../Data/test3.mhd")
    f = open('result/result2.txt', 'w')
    f_time_mean = open('result/timeAndMean2.txt', 'w')
    V0 = [0.77, -0.633, 0.0]

    start_point = start_point.split(' ')
    start_point.reverse()
    end_point = end_point.split(' ')
    end_point.reverse()
    for i in range(3):
        start_point[i] = int(start_point[i])
        end_point[i] = int(end_point[i])
    oldnorm = np.array([end_point]) - np.array([start_point])

    x_space = spac[1]
    y_space = spac[2]
    scale1 = np.array([[spac[0], 0.0, 0.0], [0.0, spac[1], 0.0], [0.0, 0.0, spac[2]]])             ###scale1 就是Λ = diag (α, β, γ) 对角矩阵
    x_scale = spac[1] / 4
    y_scale = spac[2] / 4
    scale2 = np.array([[x_scale, 0.0, 0.0], [0.0, y_scale, 0.0], [0.0, 0.0, 1.0]])                 ###scale2 就是Λ′ = diag(α′, β′, 1) 平面缩放比例
    width = 120                                                                                     ### 生成图像长宽，可调为50
    height = 120
    current_i = 0
    fina_x = []
    S = []
    begin = 0

    current_A = []
    current_A = np.copy(V0)                                ####current_A 浮点型
    current_A = start_point
    step = 0.25
    maxvec = [[]]

    time_start = time.time()
    while MethodOur.function.stop(np.array(current_A), end_point, 15):
        y = []
        maxvec[0] = np.copy(V0)
        maxvec = np.array(maxvec)
        if oldnorm.dot(maxvec.T) < 0:
            maxvec = -maxvec
        normalMat = MethodOur.function.calcNorm(maxvec.tolist()[0])
        sample = np.array(current_A)
        meanVal = sample.dot(scale1)
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
                ans = int(round(MethodOur.function.calcScale(pos.tolist(), ct)))
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
        image_pre = np.copy(image)

        for i in range(len(image)):
            for j in range(len(image[i])):
                if image[i][j] != 255:
                    image[i][j] = 0
        img = np.uint8(image)
        imgray = cv2.Canny(img, 600, 100, 3)#Canny边缘检测，参数可更改  
        ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#contours为轮廓集，可以计算轮廓的长度、面积等  
        if len(contours) == 0:
            break
        cnt = contours[0]
        S1 = cv2.contourArea(cnt)
        ell = cv2.fitEllipse(cnt)
        x = ell[0][0]
        y = ell[0][1]
        a = ell[1][1] / 2
        b = ell[1][0] / 2
        theta = ell[2]
        alpha = math.acos(math.sqrt((1 + ((b * b) / (a * a)) * math.tan(theta) * math.tan(theta)) / (1 + math.tan(theta) * math.tan(theta))))
        beta = math.acos((b / a) * math.sqrt((1 + math.tan(theta) * math.tan(theta)) / (1 + ((b * b) / (a * a)) * math.tan(theta) * math.tan(theta))))
        if V0[0] < 0:
            V0[0] = -math.cos(alpha) * math.cos(beta)
        else:
            V0[0] = math.cos(alpha) * math.cos(beta)
        if V0[1] < 0:
            V0[1] = -math.cos(alpha) * math.sin(beta)
        else:
            V0[1] = math.cos(alpha) * math.sin(beta)
        if V0[2] < 0:
             V0[2] = -math.sin(alpha)
        else:
             V0[2] = math.sin(alpha)
        current_A[0] = current_A[0] + V0[0]
        current_A[1] = current_A[1] + V0[1]
        current_A[2] = current_A[2] + V0[2]
        oldNorm = maxvec.copy()
        for i in range(3):
            current_A[i] = int(round(current_A[i]))
        # print(V0)
        print('{0} {1} {2}'.format(current_A, np.sum(image_pre) / 255 * x_scale * y_scale, current_i))
        f.write('{0}\n'.format(np.sum(image_pre) / 255 * x_scale * y_scale))

        fig, (ax0) = plt.subplots(1, 1)
        ax0.imshow(image_pre)
        plt.show()

        if current_i >= 50:
            image = np.uint8(image)
            print('fail')
            break
        current_i += 1
    time_end = time.time()
    f_time_mean.write('cost time is :{0},mean of the cost time is :{1} \n'.format(time_end-time_start, (time_end-time_start)/current_i))
    f.close()
    f_time_mean.close()






if __name__ == '__main__':
    main()
