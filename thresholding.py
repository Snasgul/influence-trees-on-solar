import cv2
import numpy as np
import sys


#np.set_printoptions(threshold=sys.maxsize)

tree_img = cv2.imread('C:/Users/patri/OneDrive/Bilder/Tree/TreeTestImg.JPG')

tree_img_hsv = cv2.cvtColor(tree_img, cv2.COLOR_BGR2HSV)

lower_bound = np.array([200, 0, 0])
upper_bound = np.array([360, 220, 220])


#tree_img_h = tree_img_hsv

#print(np.where(tree_img_h[:, :, 1] < 30, 0, tree_img_h[:, :, 1]))

#print(tree_img_h[:, :, 0])

#row_index = 0

#for row in tree_img_hsv:
#    pixel_index = 0
#    for pixel in row:
#        if pixel[1] < 60 or pixel[2] < 60:
#            tree_img_h[row_index][pixel_index] = np.array([0, 0, 0])
#        else:
#            tree_img_h[row_index][pixel_index] = pixel
#        pixel_index += 1
#    row_index += 1

#print(tree_img_h)

#tree_img_h[:, :, 1] = 0
#tree_img_h[:, :, 2] = 0

#print(tree_img_h)

#mask = np.where(tree_img_h > 90, 255, 0)



imagemask = cv2.inRange(tree_img, lower_bound, upper_bound)

#cv2.imwrite("C:/Users/patri/OneDrive/Bilder/Tree/maskedTree.jpg", tree_img_h)

cv2.imwrite("C:/Users/patri/OneDrive/Bilder/Tree/maskedTreebw.jpg", imagemask)