import cv2
import numpy as np

# 边缘检测
img = cv2.imread('datasets/image/G0001.jpg')

img_down = cv2.pyrDown(img)  # 向下采样
img_up = cv2.pyrUp(img_down)  # 向上采样
img_laplacian = img - img_up  # 拉普拉斯金字塔

cv2.imwrite('G0001.jpg', img_laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()
