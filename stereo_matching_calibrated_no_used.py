import calibrated_img as cali
import time
import cv2
import numpy as np
import os
import copy

recordTime = time.time()
startTime = time.strftime("%H%M%S")

folderName = 'LocalResult' + '_rectified_Img'
if not os.path.exists(folderName):
    os.makedirs(folderName)

fieldNum = 3
field_0 = [56, 104, 135, 46]
field_1 = [325, 48, 320, 157]
field_2 = [613, 76, 158, 76]
fieldBottom = [field_0, field_1, field_2]

boundary_fieldBottom_2_trans = [[100, 70, 140, 30], [370, 30, 340, 100], [660, 70, 170, 50]]

test = cali.calibrated_image('Photo/wide-left-rectified-8.jpg', 'Photo/wide-right-rectified-8.jpg',
                             folderName, boundary_fieldBottom_2_trans, 0.01, 0)

pyrNum = 2
target_img_pyramid, template_img_pyramid = test.build_image_pyramid(pyrNum)
'''
This parameter a, b, c is for the H from dst(empty canvas) to target img, so it's from template to target image. \
Here from right to left image
'''
H_inf = np.eye(3)
e = np.array([[1], [0], [0]], dtype=float)

fieldTop = copy.deepcopy(fieldBottom)
for i in range(fieldNum):
    for j in range(4):
        fieldTop[i][j] = int(fieldTop[i][j] / 2 ** pyrNum)

a_0, b_0, c_0 = 0, 0, 10  # test.calulate_paramater(field[0])
a_1, b_1, c_1 = 0, 0, 10  # test.calulate_paramater(field[1])
a_2, b_2, c_2 = 0, 0, 10  # test.calulate_paramater(field[2])
qTrans = [np.array([a_0, b_0, c_0]).reshape(1, 3), np.array([a_1, b_1, c_1]).reshape(1, 3),
          np.array([a_2, b_2, c_2]).reshape(1, 3)]

qTrans, qConvergeSyb = test.q_update(fieldTop, pyrNum, fieldNum, target_img_pyramid, template_img_pyramid, qTrans, e,
                                     H_inf, 1)

qTrans_final, qConvergeSyb = test.q_update(fieldBottom, pyrNum, fieldNum, target_img_pyramid, template_img_pyramid,
                                           qTrans, e, H_inf, 0)

for j in range(fieldNum):
    name = folderName + '/warped_image_of_' + 'patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '.png'
    ROI_of_template = folderName + '/patch_' + str(j + 1) + '_ROI_of_template.png'
    ROI_of_target = folderName + '/patch_' + str(j + 1) + '_ROI_of_warped_image.png'
    name_of_difference = folderName + '/patch_' + str(j + 1) + '_difference.png'

    warped_image = test.warp_image(qTrans_final[j], e, H_inf, target_img_pyramid[0], template_img_pyramid[0])
    difference = abs(
        warped_image[fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
        fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])] -
        template_img_pyramid[0][fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
        fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])

    cv2.imwrite(ROI_of_template, template_img_pyramid[0][fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                                 fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
    cv2.imwrite(ROI_of_target, warped_image[fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                               fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
    cv2.imwrite(name, warped_image)
    cv2.imwrite(name_of_difference, difference)

timeGap = time.time() - recordTime
if timeGap >= 1:
    recordTime += timeGap
    showTime_String = test.get_lapseTime(startTime, time.strftime("%H%M%S"))
    print('run time :', showTime_String)
