import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

filename = input("Filename: ")
tree_img = cv2.imread('measuredTrees/' + filename)

try:
    os.mkdir("compare/" + filename[:-4])
except FileExistsError:
    pass

sun_x = int(input("Sun position X: "))#1517
sun_y = int(input("Sun position Y: "))#1467

min_transmission = float(input("Transmission min: "))
max_transmission = float(input("Transmission max: "))

distance = 5
pos_az = 10

variation = 200

sun_radius_deg = 0.533

res_x = tree_img.shape[1]
res_y = tree_img.shape[0]
pix_deg = res_x / 65


sun_radius = int(sun_radius_deg * pix_deg) // 2

pos_x = np.arange(sun_x-sun_radius*4, sun_x+sun_radius*4, int(sun_radius/4))
pos_y = np.arange(sun_y-sun_radius*4, sun_y+sun_radius*4, int(sun_radius/4))

coords = np.array(np.meshgrid(pos_x, pos_y, )).T.reshape(-1, 2)


circle_mask = cv2.circle(np.zeros((sun_radius*2+1,sun_radius*2+1), dtype=int), (sun_radius,sun_radius), sun_radius, 255, -1)

for i in range(100, 250, 5):
    lower_bound = np.array([i, 0, 0])
    upper_bound = np.array([360, 255, 255])

    mask = cv2.inRange(tree_img, lower_bound, upper_bound)

    cv2.imwrite("compare/" + filename[:-4] + "/mask_" + str(i) + ".png", mask)

    cropped_tree = np.array([np.repeat([mask[0:sun_radius*2+1, 0:sun_radius*2+1]], pos_y.size, axis=0)])
    for x in pos_x:
        cropped_y = np.array([mask[0:sun_radius*2+1, 0:sun_radius*2+1]])
        for y in pos_y:
            cropped_y = np.append(cropped_y, [mask[y-sun_radius:y+sun_radius+1, x-sun_radius:x+sun_radius+1]], axis=0)
        cropped_y = np.delete(cropped_y, 0, axis=0)
        cropped_tree = np.append(cropped_tree, [cropped_y], axis=0)
    cropped_tree = np.delete(cropped_tree, 0, axis=0)


    masked_tree = cropped_tree * circle_mask // 65025


    lai = (np.apply_over_axes(np.sum, masked_tree, [2,3]) / (np.sum(circle_mask) // 255)).reshape(len(masked_tree),-1)


    x = np.linspace(-(len(lai)-1)/2, (len(lai)-1)/2, len(lai))
    y = x

    X, Y = np.meshgrid(x, y)

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, lai, rstride=1, cstride=1, cmap='viridis')

    plt.savefig("compare/" + filename[:-4] + "/3d_" + str(i) + ".png")
    plt.close()


    matched_values = np.logical_and(lai >= min_transmission, lai <= max_transmission)

    plt.imshow(matched_values, cmap="gray")
    plt.savefig("compare/" + filename[:-4] + "/match_" + str(i) + ".png")