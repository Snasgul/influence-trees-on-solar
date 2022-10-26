import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from astropy.coordinates import get_sun, AltAz, EarthLocation
from astropy.time import Time
import numpy as np
from astropy import units as u


def getSunPos():
    lat = 46.8768628
    lon = 7.6212138
    start = Time("2022-10-25") + 8 * u.day
    
    duration = 12 * 3600
    
    samples = 6

    dTimeArr = np.arange(0, duration, 3600 / samples) * u.hour / 3600
    timeArr = start + dTimeArr

    loc = EarthLocation(lon=lon*u.deg, lat=lat*u.deg)
    altaz = AltAz(obstime=timeArr, location=loc)
    
    sunPos = get_sun(timeArr).transform_to(altaz)

    return dTimeArr, sunPos.az, sunPos.alt

tree_img = cv2.imread('C:/Users/patri/OneDrive/Bilder/Tree/IMG_0671.JPG')
plt.imshow(tree_img)

distance = 5
pos_az = 10

img_az = 135
img_alt = 43

sun_pos = getSunPos()

sun_az = sun_pos[1]
sun_alt = sun_pos[2]
sun_size_deg = 0.533

res_x = tree_img.shape[1]
res_y = tree_img.shape[0]
pix_deg = res_x / 65

print(res_y)

lower_bound = np.array([200, 0, 0])
upper_bound = np.array([360, 220, 220])

mask = cv2.inRange(tree_img, lower_bound, upper_bound)

#cv2.imshow('treemask', imagemask)
plt.imshow(mask, cmap='gray')

cv2.imwrite("C:/Users/patri/OneDrive/Bilder/Tree/test_tree_gray.jpg", mask)

sun_az_rel = sun_az - img_az
sun_alt_rel = sun_alt - img_alt

center_x = sun_az_rel * pix_deg
center_y = sun_alt_rel * pix_deg

pos_x = int(res_x / 2 + center_x)
pos_y = int(res_y / 2 - center_y)

sun_size = int(sun_size_deg * pix_deg)
print(sun_size)

circle_mask = np.zeros_like(mask)
circle_mask = cv2.circle(circle_mask, (pos_x,pos_y), sun_size, 255, -1)

plt.imshow(circle_mask, cmap='gray')

tree_cut = mask * circle_mask

plt.imshow(tree_cut, cmap='gray')

plt.imshow(tree_cut[pos_y-sun_size:pos_y+sun_size+1, pos_x-sun_size:pos_x+sun_size+1], cmap='gray')

lai = np.sum(tree_cut) / np.sum(circle_mask) * 255
print(lai)


