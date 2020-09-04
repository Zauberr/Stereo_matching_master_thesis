import time

import calibrated_img as cali

import numpy as np
import cv2

import copy
import os


recordTime = time.time()
startTime = time.strftime("%H%M%S")

# three target area in bottom(with small label) layer of pyramid for template_img
'''
first patch:(Billboard on the left balcony)
        Rectangle Selection
            X: 56
            Y: 135
            Width: 104
            Height: 46
second patch:(Billboard on the left wall)
        Rectangle Selection
            X: 325
            Y: 320
            Width: 48
            Height: 157
third patch:(Text on the right wall)
        Rectangle Selection
            X: 613
            Y: 158
            Width: 76
            Height: 76
Rectangle Selection 3
  X: 755
  Y: 374
  Width: 45
  Height: 161
In form [x, width, y, height]
'''
fieldNum = 4
field_0 = [35, 145, 135, 48]
field_1 = [325, 48, 320, 157]
field_2 = [569, 240, 167, 66]
field_3 = [755, 45, 374, 161]
fieldBottom = [field_0, field_1, field_2, field_3]

"""
build a boundary for the three field in target_img.
but for this, if we need a color boundary, we have to ues the color image.
until now we only implement the algorithm for the gray image. So now we first use a boundary in white.
Then for the next step, I will extend my algorithm to the color image and use a color boundary
"""
boundary_fieldBottom_2_trans = [[100, 70, 140, 30], [370, 30, 340, 100], [660, 70, 170, 50]]

# threshold definition
qThreshold = 1e-3
heThreshold = 1e-3

# save the image in a folder
folderName = 'globalResult_' + str(qThreshold) + '_' + str(heThreshold) + 'for_image_with_white_house'
if not os.path.exists(folderName):
    os.makedirs(folderName)

# left as target image, right as template image. H transform template to target(right to left)
test = cali.calibrated_image('Photo/wide-left-rectified-8.jpg', 'Photo/wide-right-rectified-8.jpg', folderName,
                             qThreshold, heThreshold)

# build the image Pyramid(pyrNum doesn't include original image)
# 0 layer is the original Image

pyrNum = 2
target_img_pyramid, template_img_pyramid = test.build_image_pyramid(pyrNum, 0)

# for calibrated_img H_{\infinity} is I and e is [[1],[0],[0]]

"""
H_inf e and q are the parameter of dehomogenization. When we warped the 2_transform_image to form_image, we have to use inv(H). 
And I have consider it in function warp_image.
"""

H_inf = np.eye(3)
e = np.array([[1], [0], [0]], dtype=float)

# build the field to top layer
fieldTop = copy.deepcopy(fieldBottom)
for i in range(fieldNum):
    for j in range(4):
        fieldTop[i][j] = int(fieldTop[i][j] / 2 ** pyrNum)

a_0, b_0, c_0 = 0, 0, 32.0
a_1, b_1, c_1 = 0, 0, 40.0
a_2, b_2, c_2 = 0, 0, 30.0
a_3, b_3, c_3 = 0, 0, 30.0
q = [np.array([a_0, b_0, c_0]).reshape(3, 1), np.array([a_1, b_1, c_1]).reshape(3, 1),
     np.array([a_2, b_2, c_2]).reshape(3, 1), np.array([a_3, b_3, c_3]).reshape(3, 1)]

a, b = 1.0, 0.0
r_correct = [np.array([a, b]).reshape(2, 1), np.array([a, b]).reshape(2, 1),
             np.array([a, b]).reshape(2, 1), np.array([a, b]).reshape(2, 1)]

max_loop_Number = 20

for i in range(max_loop_Number):

    print("loop Number:", i + 1)

    # calculate q for different patches, pyramidOrNot = 0 means only calculating the original layer.
    qTemporary, r_correctTemp, qConvergeSyb = test.q_update(fieldBottom, pyrNum, fieldNum, target_img_pyramid[1],
                                                            template_img_pyramid[1], q, e, H_inf, r_correct, 0, 1)

    # calculate E and epsilon
    H_inf_temporary, e_temporary, HeConvergeSyb = test.H_inf_and_e_update(fieldNum, fieldBottom, qTemporary, e, H_inf,
                                                                          r_correctTemp, target_img_pyramid[1],
                                                                          template_img_pyramid[1])

    # show result for every cycle
    normDelta_q = 0

    for j in range(fieldNum):
        name = folderName + '/warped_image_of_' + 'patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '.png'
        ROI_of_template = folderName + '/patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '_ROI_of_template.png'
        ROI_of_target = folderName + '/patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '_ROI_of_warped_image.png'
        name_of_difference = folderName + '/patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '_difference.png'

        warped_image = test.warp_image(qTemporary[j], e_temporary, H_inf_temporary, target_img_pyramid[1],
                                       template_img_pyramid[1])
        difference, legend = test.show_difference_of_warp_image(warped_image, template_img_pyramid[1], fieldBottom[j],
                                                                5)

        cv2.imwrite(ROI_of_template, template_img_pyramid[1][fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
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

# change the q to bottom layer bigger

# for j in range(fieldNum):
#     q_temporary[j][2] = q_temporary[j][2] * 4


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
