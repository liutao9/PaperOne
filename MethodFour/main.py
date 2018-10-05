import function
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import math
from skimage import draw, transform, feature


#
# 该算法存在的缺点：
# 1，特别依赖于初始化值。如果起始点和方向向量初始化不正确，会导致程序出错。主要原因是该算法使用两个点作为方向向量，这样会存在很大误差
# 然而算法的自动校正的能力几乎为零，从而会导致误差越来越大，甚至出错。
# 2，算法依赖于质心检测。程序中使用基于霍夫变换的圆形检测，将圆心作为质心，如果质心位置不合适，直接影响了方向向量和下一个起始点。
# 3，程序对算法做了修改。直接将计算出的pos位置作为下一个起始点，这样会存在较大误差，我们将程序中的方向向量作为了起始点的校正向量，减小移动幅度，从而减小
# 程序误差，提高程序健壮性。

def main():
    #'''
    #圆环面积测试
    #start_point = '64 108 110'
    start_point = '64 105 108'
    end_point = '66 102 144'
    ct, ori, spac = function.load_itk("../Data/test2.mhd")
    f = open('result/result.txt', 'w')
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

    V0 = [0.77, -0.633, 0.0]
    current_A = []
    current_B = []
    current_A = np.copy(start_point)
    current_B = np.copy(start_point)
    step = 0.25
    maxvec = [[]]

    while function.stop(np.array(current_A), end_point, 15):
        y = []
        maxvec[0] = np.copy(V0)
        maxvec[0][0] = V0[0] / (np.sqrt(V0[0] * V0[0] + V0[1] * V0[1] + V0[2] * V0[2]))
        maxvec[0][1] = V0[1] / (np.sqrt(V0[0] * V0[0] + V0[1] * V0[1] + V0[2] * V0[2]))
        maxvec[0][2] = V0[2] / (np.sqrt(V0[0] * V0[0] + V0[1] * V0[1] + V0[2] * V0[2]))
        maxvec = np.array(maxvec)
        if oldnorm.dot(maxvec.T) < 0:
            maxvec = -maxvec
        current_B[0] = current_A[0] + step * maxvec[0][0]
        current_B[1] = current_A[1] + step * maxvec[0][1]
        current_B[2] = current_A[2] + step * maxvec[0][2]
        ### 根据点current_B 和 切向量 V0 计算出平面C1
        normalMat = function.calcNorm(maxvec.tolist()[0])
        sample = np.array(current_B)
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
        ### 计算出 圆心，  将current_A和圆心形成的向量作为新的V0
        ####start_point = (meanVal + step * np.linalg.inv(scale1).dot(maxvec.T).T).tolist()[0]
        img = np.uint8(image)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 600, param1=100, param2=1)
        circles = np.uint8(np.around(circles))
        circle = circles[0][0]
        pos = transMat.dot(np.array([x_scale * circle[0], y_scale * circle[1], 0.0, 1.0]))
        V0[0] = pos[2] - current_A[2]               ### 这有点问题，将(x,y,z)顺序 搞成了 (z,y,x)
        V0[1] = pos[1] - current_A[1]# - 1           ### 可能是精度问题，并没有按照我们想要的方向走
        V0[2] = pos[0] - current_A[0]
        #current_A = pos[0:3]           ###这样直接赋值，幅度太大，应将方向向量V0作为current_A的校正向量
        current_A[0] = current_A[0] + V0[0]
        current_A[1] = current_A[1] + V0[1]
        current_A[2] = current_A[2] + V0[2]
        print('{0} {1} {2}'.format(current_A, np.sum(image) / 255 * x_scale * y_scale, current_i))
        print("pos:{0} current_A:{1} V0: {2} current_B: {3}\n".format(pos, current_A, V0, current_B))

        oldNorm = maxvec.copy()
        for i in range(3):
            current_A[i] = int(round(current_A[i]))

        #
        # fig, (ax0) = plt.subplots(1, 1)
        # ax0.imshow(image)
        # plt.show()

        if current_i >= 50:
            image = np.uint8(image)
            print('fail')
            for i in range(width):
                for j in range(height):
                    f.write('%.3d ' % (image[i, j]))
                f.write('\n')
            break
        current_i += 1
    f.close()






if __name__ == '__main__':
    main()
