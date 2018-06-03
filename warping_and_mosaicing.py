#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import os

import matplotlib.pyplot as plt

__author__ = 'Nicolás Cerna'

# PARAMETERS ##################################################################

ransac_max_dist = 2  # distance for RANSAC to classify a point as an inliner
n = 11  # number of points to calculate the homography

cwd = os.path.dirname(os.path.abspath(__file__))  # current script path
figures_path = os.path.join(cwd, 'figures')

img_to_warp_path = os.path.join(figures_path, '1.jpg')
img_base_path = os.path.join(figures_path, '2.jpg')

img_base = cv2.imread(img_base_path, 0)  # this image will not be modified
img_to_warp = cv2.imread(img_to_warp_path, 0)  # this image will be warped

img_base_rgb = cv2.cvtColor(cv2.imread(img_base_path, cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB)
img_to_warp_rgb = cv2.cvtColor(cv2.imread(img_to_warp_path, cv2.IMREAD_COLOR),
                               cv2.COLOR_BGR2RGB)
# THE ALGORITHM ###############################################################

# creates the ORB detector (depending on the OpenCV version this varies)
if int(cv2.__version__.replace('.', '')) > 330:

    orb = cv2.ORB_create()

else:

    orb = cv2.ORB()

# finds the key points and their respective descriptors using ORB
kp1, des1 = orb.detectAndCompute(img_base, None)
kp2, des2 = orb.detectAndCompute(img_to_warp, None)

# creates the matcher that will be used to find the shortest distance between
# descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# finds the matches between descriptors
matches = bf.match(des1, des2)

# sorts the matches using the distance between the pairs
matches = sorted(matches, key=lambda mt: mt.distance)
best_matches = matches[:n]

# finds the source points and the destiny points required to calculate the
# homography
src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches])
src_pts = src_pts.reshape(-1, 1, 2)

dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches])
dst_pts = dst_pts.reshape(-1, 1, 2)

# calculates the homography matrix using the mask given by the RANSAC algorithm
M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_max_dist)
M = M / M[2, 2]
H = np.linalg.inv(M)

# now we need to calcluate a canvas so that both images
# (the warped and the base image) could be placed inside it without losing
# information
pt1 = np.ones(3, np.float32)
pt2 = np.ones(3, np.float32)
pt3 = np.ones(3, np.float32)
pt4 = np.ones(3, np.float32)
(y, x) = img_base.shape[:2]

pt1[:2] = [0, 0]
pt2[:2] = [x, 0]
pt3[:2] = [0, y]
pt4[:2] = [x, y]

max_x = None
max_y = None
min_x = None
min_y = None

for pt in [pt1, pt2, pt3, pt4]:

    hp = np.matrix(H, np.float32) * np.matrix(pt, np.float32).T
    hp_arr = np.array(hp, np.float32)
    normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]],
                         np.float32)
    if max_x is None or normal_pt[0, 0] > max_x:
        max_x = normal_pt[0, 0]
    if max_y is None or normal_pt[1, 0] > max_y:
        max_y = normal_pt[1, 0]
    if min_x is None or normal_pt[0, 0] < min_x:
        min_x = normal_pt[0, 0]
    if min_y is None or normal_pt[1, 0] < min_y:
        min_y = normal_pt[1, 0]

min_x = min(0, min_x)
min_y = min(0, min_y)

max_x = max(max_x, img_base.shape[1])
max_y = max(max_y, img_base.shape[0])

# calculates the translation matrix
T = np.matrix(np.identity(3), np.float32)

if min_x < 0:
    T[0, 2] += -min_x
    max_x += -min_x

if min_y < 0:
    T[1, 2] += -min_y
    max_y += -min_y

img_w = int(math.ceil(max_x))
img_h = int(math.ceil(max_y))

# performs the translation of the base image (in color) to perform the
# homography correctly
img_base_translated = cv2.warpPerspective(img_base_rgb, T, (img_w, img_h))

# calculates the new transformation incorporating the translation
M_inv = T * np.linalg.inv(M)

# warps the new image given the homography
warped_img = cv2.warpPerspective(img_to_warp_rgb, M_inv, (img_w, img_h))

# canvas where the images will be added
canvas = np.zeros((img_h, img_w, 3), np.uint8)

# creates a mask from the warped image
_, mask = cv2.threshold(warped_img, 0, 255, cv2.THRESH_BINARY_INV)

# adds the images using the mask (we used the mask of the R channel)
pre_final_img = cv2.add(canvas, img_base_translated, mask=mask[:, :, 0],
                        dtype=cv2.CV_8U)

# creates the final image adding pre_final_img to img_base
img_final = cv2.add(pre_final_img, warped_img, dtype=cv2.CV_8U)

# shows the result
plt.figure()
plt.imshow(img_final)
plt.title('Result')

cv2.imwrite(os.path.join(figures_path, 'result.jpg'),
            cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
