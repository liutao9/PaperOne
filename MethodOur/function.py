import SimpleITK as sitk
import numpy as np
import math

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return ct_scan, origin, spacing

def calcNorm(norm):                                                                              ###calcNorm 公式2-6中 中间的3*3矩阵
    r = norm[0] * norm[0] + norm[1] * norm[1]
    if r == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    else:
        return np.array([[-norm[1] / r, -norm[0] * norm[2] / r, norm[0]], [norm[0] / r, -norm[1] * norm[2] / r, norm[1]], [0.0, r, norm[2]]])

def dist(pos, x, y, z):
    return math.sqrt((pos[0] - x) * (pos[0] - x) + (pos[1] - y) * (pos[1] - y) + (pos[2] - z) * (pos[2] - z))

def calcScale(pos, image):
    x = int(pos[0])
    y = int(pos[1])
    z = int(pos[2])
    r = []
    gray = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                di = dist(pos, x + i, y + j, z + k)
                if di == 0:
                    return image[x + i][y + j][z + k]
                r.append(1.0 / di)
                gray += 1.0 / di * image[x + i][y + j][z + k]
    gray /= sum(r)
    return gray

def stop(x, end, num):
    r = math.sqrt((x - end).dot((x - end).T))
    if r <= num:
        return False
    return True
