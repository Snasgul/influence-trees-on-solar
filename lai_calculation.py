import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

tree_img = cv2.imread('C:/Users/patri/OneDrive/Bilder/Tree/TreeTestImg.JPG')

lower_bound = np.array([200, 0, 0])
upper_bound = np.array([360, 220, 220])

mask = cv2.inRange(tree_img, lower_bound, upper_bound)

#plt.imshow(mask, cmap='gray')
#plt.show()

circle_mask = np.zeros_like(mask)
circle_mask = cv2.circle(circle_mask, (300,200), 100, 255, -1)

#plt.imshow(circle_mask, cmap='gray')
#plt.show()

tree_cut = mask * circle_mask

#plt.imshow(tree_cut, cmap='gray')
#plt.show()

lai = np.sum(tree_cut) / np.sum(circle_mask) * 255

print(lai)


