import time

import calibrated_img as cali

import numpy as np
import cv2

import copy
import os

recordTime = time.time()
startTime = time.strftime("%H%M%S")

# save the image in a folder
folderName = 'estimation_result'
if not os.path.exists(folderName):
    os.makedirs(folderName)

# save the result in txt.file
text_name = folderName + '/RMSD.txt'
text = open(text_name, 'w+')

# three target area in bottom(with small label) layer of pyramid for left_img(template_img 1)
'''
Rectangle Selection
  X: 368
  Y: 310
  Width: 209
  Height: 168
Rectangle Selection
  X: 674
  Y: 295
  Width: 156
  Height: 194
Rectangle Selection
  X: 483
  Y: 582
  Width: 216
  Height: 114
In form [x, width, y, height]
'''
fieldNum = 3
field_0 = [368, 209, 310, 168]
field_1 = [674, 156, 295, 194]
field_2 = [483, 216, 582, 114]
fieldBottom = [field_0, field_1, field_2]

# threshold definition
qThreshold = 1e-4
heThreshold = 1e-5

# left as target image, right as template image. H transform template to target(right to left)
test = cali.calibrated_image('R.png', 'L.png', folderName, qThreshold, heThreshold)

H_inf_gt, e_gt, q_gt = test.read_ground_truth('ground_truth_number.txt')
H_inf_temporary, e_temporary, qTemporary, r_temporary = test.read_estimation_result(
    '/Users/zauber/Desktop/Stereo_matching/Built_stereo_image/Normal_case_1/estimation_result/Estimation_number.txt')
# calculate RMSD and maximum displacement
RMSD, d_max, RMSD_each, d_max_each = test.calculate_RMSD(H_inf_gt, e_gt, q_gt, H_inf_temporary, e_temporary, qTemporary,
                                                         fieldNum,
                                                         fieldBottom)

print('RMSD:', RMSD)
print('RMSD:', file=text)
print(RMSD, file=text)

print('RMSD_each:', RMSD_each)
print('RMSD_each:', file=text)
print(RMSD_each, file=text)

print('d_max:', d_max)
print('d_max:', file=text)
print(d_max, file=text)

print('d_max_each:', d_max_each)
print('d_max_each:', file=text)
print(d_max_each, file=text)

text.close()

timeGap = time.time() - recordTime
if timeGap >= 1:
    recordTime = timeGap
    showTime_String = test.get_lapseTime(startTime, time.strftime("%H%M%S"))
    print('run time :', showTime_String)
