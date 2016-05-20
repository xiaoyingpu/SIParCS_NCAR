import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("pacific.jpg", 0)
edges = cv2.Canny(img, 10, 20)
plt.subplot(121), plt.imshow(img,cmap = 'gray')
plt.subplot(122), plt.imshow(edges,cmap = 'gray')
plt.show()

