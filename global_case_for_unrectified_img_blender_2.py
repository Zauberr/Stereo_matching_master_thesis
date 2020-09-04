import time

import calibrated_img as cali

import numpy as np
import cv2

import copy
import os

recordTime = time.time()
startTime = time.strftime("%H%M%S")

# three target area in bottom(with small label) layer of pyramid for template_img(here right)
# H is from template to target_img(here right to left)
'''
first patch:brick wall
Rectangle Selection
  X: 768
  Y: 142
  Width: 260
  Height: 240
-40
second patch: color wall
Rectangle Selection
  X: 1226
  Y: 240
  Width: 214
  Height: 242
-40
third patch: Markov wall
Rectangle Selection
  X: 824
  Y: 608
  Width: 290
  Height: 156
-30
In form [x, width, y, height]
'''
fieldNum = 3
field_0 = [768, 260, 142, 240]
field_1 = [1226, 214, 240, 242]
field_2 = [824, 290, 608, 156]
fieldBottom = [field_0, field_1, field_2]

# threshold definition
qThreshold = 1e-3
heThreshold = 1e-3

# save the image in a folder
folderName = 'globalResult_' + 'Test_img_from_Blender_' + str(qThreshold) + '_' + str(heThreshold)
if not os.path.exists(folderName):
    os.makedirs(folderName)

test = cali.calibrated_image('Photo/Test_Image_from_Blender/First_test_without_groundtruth_left.png',
                             'Photo/Test_Image_from_Blender/First_test_without_groundtruth_right.png',
                             folderName, qThreshold, heThreshold)

# build the image Pyramid(pyrNum doesn't include original image)
# 0 layer is the original Image

pyrNum = 2
target_img_pyramid, template_img_pyramid = test.build_image_pyramid(pyrNum)

H_inf = np.eye(3)
e = np.array([[1], [0], [0]], dtype=float)

# build the field to top layer
fieldTop = copy.deepcopy(fieldBottom)
for i in range(fieldNum):
    for j in range(4):
        fieldTop[i][j] = int(fieldTop[i][j] / 2 ** pyrNum)

a_0, b_0, c_0 = 0, 0, -40
a_1, b_1, c_1 = 0, 0, -40
a_2, b_2, c_2 = 0, 0, -30
q = [np.array([a_0, b_0, c_0]).reshape(3, 1), np.array([a_1, b_1, c_1]).reshape(3, 1),
     np.array([a_2, b_2, c_2]).reshape(3, 1)]

max_loop_Number = 20

a, b = 1, 0
r_correct = [np.array([a, b]).reshape(2, 1), np.array([a, b]).reshape(2, 1),
             np.array([a, b]).reshape(2, 1)]

for i in range(max_loop_Number):

    print("loop Number:", i + 1)

    # calculate q for different patches, pyramidOrNot = 0 means only calculating the original layer.
    qTemporary, r_correctTemp, qConvergeSyb = test.q_update(fieldBottom, pyrNum, fieldNum, target_img_pyramid,
                                                            template_img_pyramid, q, e, H_inf, r_correct, 0)

    # calculate E and epsilon
    H_inf_temporary, e_temporary, HeConvergeSyb = test.H_inf_and_e_update(fieldNum, fieldBottom, qTemporary, e, H_inf,
                                                                          r_correctTemp, target_img_pyramid,
                                                                          template_img_pyramid)
    # show result for every cycle

    normDelta_q = 0

    for j in range(fieldNum):
        name = folderName + '/warped_image_of_' + 'patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '.png'
        ROI_of_template = folderName + '/patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '_ROI_of_template.png'
        ROI_of_target = folderName + '/patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '_ROI_of_warped_image.png'
        name_of_difference = folderName + '/patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '_difference.png'

        warped_image = test.warp_image(qTemporary[j], e_temporary, H_inf_temporary, target_img_pyramid[0],
                                       template_img_pyramid[0])

        difference = test.show_difference_of_warp_image(warped_image, template_img_pyramid[0], fieldBottom[j])

        cv2.imwrite(ROI_of_template, template_img_pyramid[0][fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                                     fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
        cv2.imwrite(ROI_of_target, warped_image[fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                                   fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
        cv2.imwrite(name, warped_image)
        cv2.imwrite(name_of_difference, difference)

    # stopping criteria
    Delta_q_sum = np.zeros((3, 1))
    Delta_r_correct_sum = np.zeros((2, 1))
    q_sum = np.zeros((3, 1))
    r_correct_sum = np.zeros((2, 1))
    for k in range(fieldNum):
        q_sum = q_sum + q[k]
        r_correct_sum = r_correct_sum + r_correct[k]
        Delta_q_sum = Delta_q_sum + qTemporary[k] - q[k]
        Delta_r_correct_sum = Delta_r_correct_sum + r_correctTemp[k] - r_correct[k]

    Delta_e = e_temporary - e
    Delta_H_inf = H_inf_temporary - H_inf

    FinalThres = 1e-2

    if np.linalg.norm(Delta_q_sum) / np.linalg.norm(q_sum) < FinalThres and np.linalg.norm(Delta_e) / np.linalg.norm(
            e) < \
            FinalThres and np.linalg.norm(Delta_H_inf) / np.linalg.norm(H_inf) < FinalThres and \
            np.linalg.norm(Delta_r_correct_sum) / np.linalg.norm(r_correct_sum) < FinalThres:
        print("the result converges")
        break

    if i == max_loop_Number - 1:
        print("Until the max loop, it doesn't converge")

    q = copy.deepcopy(qTemporary)
    H_inf = H_inf_temporary
    e = e_temporary
    r_correct = r_correctTemp

print('Final Result:')
print('H_inf:', H_inf_temporary)
print('e:', e_temporary)
print('q:', qTemporary)
print('r_correct:', r_correctTemp)

timeGap = time.time() - recordTime
if timeGap >= 1:
    recordTime += timeGap
    showTime_String = test.get_lapseTime(startTime, time.strftime("%H%M%S"))
    print('run time :', showTime_String)
