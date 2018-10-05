import MethodOur.function
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import time

def main():
    #'''
    #真实的血管
    start_point = '123 204 98'
    end_point = '141 218 82'
    ct, ori, spac = MethodOur.function.load_itk("../Data/Confidence_Connected_Yi_Bao_Jun_4.mhd")
    V0 = [-0.67479813, 0.31554728, 0.66714121]
    f = open('result/result3.txt', 'w')
    f_time_mean = open('result/timeAndMean3.txt', 'w')
    #'''
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
    width = 60                                                                                     ### 生成图像长宽，可调为50
    height = 60

    current_A = np.copy(V0)                                ####current_A 浮点型
    current_A = start_point
    maxvec = [[]]

    h = 10
    current_V = np.copy(V0)
    count = 0

    time_start = time.time()
    while MethodOur.function.stop(np.array(current_A), end_point, 15):

        max_sum = 0
        best_V = np.copy(V0)
        current_i = 0

        for current_alpha in np.arange(0.0, 0.2 * math.pi, math.pi / 10.0):
            for current_beta in np.arange(0.0, 2.0 * math.pi, math.pi / 10.0):
                current_V[0] = V0[0] + math.cos(current_alpha) * math.sin(current_beta) * 0.1
                current_V[1] = V0[1] + math.cos(current_alpha) * math.cos(current_beta) * 0.1
                current_V[2] = V0[2] + math.sin(current_alpha) * 0.1
                current_X = np.cos(current_A)
                current_sum = 0


                for current_h in range(h):
                    current_X[0] = current_A[0] + current_V[0] * current_h
                    current_X[1] = current_A[1] + current_V[1] * current_h
                    current_X[2] = current_A[2] + current_V[2] * current_h

                    maxvec[0] = np.copy(current_V)
                    maxvec = np.array(maxvec)
                    if oldnorm.dot(maxvec.T) < 0:
                        maxvec = -maxvec
                    normalMat = MethodOur.function.calcNorm(maxvec.tolist()[0])
                    sample = np.array(current_X)
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
                    image = cv2.GaussianBlur(image, (5, 5), 1.5)
                    # fig, (ax0) = plt.subplots(1, 1)
                    # ax0.imshow(image)
                    # plt.show()

                    current_sum += np.sum(image) / 255 * x_scale * y_scale

                current_i += 1
                #print("{0}, {1}, {2},{3},{4}".format(current_sum, current_i, current_V, current_alpha, current_beta))
                if current_sum > max_sum:
                    max_sum = current_sum
                    best_V = np.copy(current_V)
        maxvec[0] = np.copy(best_V)
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
        image = cv2.GaussianBlur(image, (5, 5), 1.5)
        count += 1
        print("the area is {0},the count is: {1}, {2},  {3}".format(np.sum(image) / 255 * x_scale, count, best_V, current_A))
        f.write('{0}\n'.format(np.sum(image) / 255 * x_scale))

        # fig, (ax0) = plt.subplots(1, 1)
        # ax0.imshow(image)
        # plt.show()

        V0 = best_V
        current_A[0] += V0[0]
        current_A[1] += V0[1]
        current_A[2] += V0[2]

    time_end = time.time()
    f_time_mean.write('cost time is :{0},mean of the cost time is :{1} \n'.format(time_end-time_start, (time_end-time_start)/count))
    f.close()
    f_time_mean.close()




if __name__ == '__main__':
    main()
