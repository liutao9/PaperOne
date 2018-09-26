import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from skimage import draw, transform, feature
from MethodOur import function

ct, ori, spac = function.load_itk("../Data/test3.mhd")
img = ct[127, :, :]
img = img[0:128, :]
result = transform.hough_ellipse(img)
result.sort(order='accumulator') #根据累加器排序

best = list(result[-1])
print(best)
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]
area = math.pi * a * b
print(area)

cy, cx = draw.ellipse_perimeter(yc, xc, a, b, orientation)
img[cy, cx] = (200)
fig2, ax = plt.subplots()
ax.imshow(img)
plt.show()
