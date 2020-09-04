import calibrated_img as cali
import time
import cv2
import numpy as np
import os
import copy

recordTime = time.time()
startTime = time.strftime("%H%M%S")

folderName = 'LocalResult' + '_rectified_bull'
if not os.path.exists(folderName):
    os.makedirs(folderName)

'''
Rectangle Selection upper left
  X: 31
  Y: 39
  Width: 189
  Height: 166

Rectangle Selection upper right
  X: 314
  Y: 53
  Width: 81
  Height: 164

Rectangle Selection below
  X: 50
  Y: 273
  Width: 238
  Height: 78

'''
fieldNum = 3
field_0 = [31, 189, 39, 166]
field_1 = [314, 81, 53, 164]
field_2 = [50, 238, 273, 78]

fieldBottom = [field_0, field_1, field_2]


# img2 as target image and img6 as template img
test = cali.calibrated_image('Photo/bull/im2.ppm', 'Photo/bull/im6.ppm',
                             folderName, 0.0001, 0)

pyrNum = 2
target_img_pyramid, template_img_pyramid = test.build_image_pyramid(pyrNum)
'''
This parameter a, b, c is for the H from template to target img, so it's from template to target image. \
Here from right to left image
'''
H_inf = np.eye(3)
e = np.array([[1], [0], [0]], dtype=float)

fieldTop = copy.deepcopy(fieldBottom)
for i in range(fieldNum):
    for j in range(4):
        fieldTop[i][j] = int(fieldTop[i][j] / 2 ** pyrNum)

a_0, b_0, c_0 = 0, 0, 5  # test.calulate_paramater(field[0])
a_1, b_1, c_1 = 0, 0, 5  # test.calulate_paramater(field[1])
a_2, b_2, c_2 = 0, 0, 10  # test.calulate_paramater(field[2])

q = [np.array([a_0, b_0, c_0]).reshape(3, 1), np.array([a_1, b_1, c_1]).reshape(3, 1),
     np.array([a_2, b_2, c_2]).reshape(3, 1)]

# parameter for radiometric correction
a, b = 1, 0
r_correct = [np.array([a, b]).reshape(2, 1), np.array([a, b]).reshape(2, 1), np.array([a, b]).reshape(2, 1)]

# qTrans, qConvergeSyb = test.q_update(fieldTop, pyrNum, fieldNum, target_img_pyramid, template_img_pyramid, qTrans, e,
#                                      H_inf, 1)

q_final, r_correct_final, qConvergeSyb = test.q_update(fieldBottom, pyrNum, fieldNum, target_img_pyramid,
                                                       template_img_pyramid,
                                                       q, e, H_inf, r_correct, 0)
print('r_correct:', r_correct_final)
for j in range(fieldNum):
    name = folderName + '/warped_image_of_' + 'patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '.png'
    ROI_of_template = folderName + '/patch_' + str(j + 1) + '_ROI_of_template.png'
    ROI_of_target = folderName + '/patch_' + str(j + 1) + '_ROI_of_warped_image.png'
    name_of_difference = folderName + '/patch_' + str(j + 1) + '_difference.png'

    warped_image = test.warp_image(q_final[j], e, H_inf, target_img_pyramid[0], template_img_pyramid[0])
    difference = test.show_difference_of_warp_image(warped_image, template_img_pyramid[0], fieldBottom[j])

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

'''
the initialization is very important, when the x,y is very large, h_1 and h_2 has more influence, 
and the result stretch. So I have to continue to work on it. And the result converge at a local minimum. 
not perfectly rectified.
'''
