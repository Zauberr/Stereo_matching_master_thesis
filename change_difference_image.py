import calibrated_img as cali

import numpy as np
import cv2

import copy
import os

old = cv2.imread('./Built_stereo_image/Normal_case_3/estimation_result/patch_1_difference.png')

height_y, width_x, color = np.array(old).shape
old_b = old[:, :, 0]
old_r = old[:, :, 2]
old = 5 * old

print(np.array(old).shape)

# for y in range(0, height_y):
#     for x in range(0, width_x):
#         if old[y, x, 0] == 0 and old[y, x, 1] == 0 and old[y, x, 2] == 0:
#             old[y, x, 0:3] = 255
cv2.imwrite('patch_1_difference_new.png', old)
