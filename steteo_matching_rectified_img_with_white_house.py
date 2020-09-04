import calibrated_img as cali
import time
import cv2
import numpy as np
import os
import copy

recordTime = time.time()
startTime = time.strftime("%H%M%S")

folderName = 'LocalResult' + '_rectified_Img_with_white_wall_unblur'
if not os.path.exists(folderName):
    os.makedirs(folderName)

'''
Rectangle Selection 0
  X: 35
  Y: 135
  Width: 145
  Height: 48
Rectangle Selection 2
  X: 569
  Y: 167
  Width: 240
  Height: 66


Rectangle Selection 3
  X: 755
  Y: 374
  Width: 45
  Height: 161
'''
fieldNum = 4
field_0 = [35, 145, 135, 48]
field_1 = [325, 48, 320, 157]
field_2 = [569, 240, 167, 66]
field_3 = [755, 45, 374, 161]
fieldBottom = [field_0, field_1, field_2, field_3]

qThreshold = 1e-4
heThreshold = 0

test = cali.calibrated_image('Photo/wide-left-rectified-8.jpg', 'Photo/wide-right-rectified-8.jpg',
                             folderName, qThreshold, heThreshold)

pyrNum = 2
target_img_pyramid, template_img_pyramid = test.build_image_pyramid(pyrNum, 0)
'''
This parameter a, b, c is for the H from template to target img, so it's from template to target image. \
Here from right to left image
'''
H_inf = np.eye(3, dtype=np.float32)
e = np.array([[1], [0], [0]], dtype=np.float32)

fieldTop = copy.deepcopy(fieldBottom)
for i in range(fieldNum):
    for j in range(4):
        fieldTop[i][j] = int(fieldTop[i][j] / 2 ** pyrNum)

a_0, b_0, c_0 = 0, 0, 32
a_1, b_1, c_1 = 0, 0, 40
a_2, b_2, c_2 = 0, 0, 30
a_3, b_3, c_3 = 0, 0, 30
q = [np.array([a_0, b_0, c_0], dtype=np.float32).reshape(3, 1),
     np.array([a_1, b_1, c_1], dtype=np.float32).reshape(3, 1),
     np.array([a_2, b_2, c_2], dtype=np.float32).reshape(3, 1),
     np.array([a_3, b_3, c_3], dtype=np.float32).reshape(3, 1)]

a, b = 1.0, 0.0
r_correct = [np.array([a, b], dtype=np.float32).reshape(2, 1), np.array([a, b], dtype=np.float32).reshape(2, 1),
             np.array([a, b], dtype=np.float32).reshape(2, 1), np.array([a, b], dtype=np.float32).reshape(2, 1)]
# blurred image
q_final_blur, r_correct_final_blur, qConvergeSyb_blur = test.q_update(fieldBottom, pyrNum, fieldNum,
                                                                      target_img_pyramid[0],
                                                                      template_img_pyramid[0], q, e, H_inf, r_correct,
                                                                      0, 1)
# original image
q_final, r_correct_final, qConvergeSyb = test.q_update(fieldBottom, pyrNum, fieldNum, target_img_pyramid[1],
                                                       template_img_pyramid[1], q_final_blur, e, H_inf,
                                                       r_correct_final_blur, 0, 1)
print('q_final:', q_final)
print('r_correct_final:', r_correct_final)

psnr = []
for j in range(fieldNum):
    name = folderName + '/warped_image_of_' + 'patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '.png'
    ROI_of_template = folderName + '/patch_' + str(j + 1) + '_ROI_of_template.png'
    ROI_of_target = folderName + '/patch_' + str(j + 1) + '_ROI_of_warped_image.png'
    name_of_difference = folderName + '/patch_' + str(j + 1) + '_difference.png'
    name_of_legend = folderName + '/patch_' + str(j + 1) + '_legend.png'
    warped_image = test.warp_image(q_final[j], e, H_inf, target_img_pyramid[1], template_img_pyramid[1])
    difference, legend = test.show_difference_of_warp_image(warped_image, template_img_pyramid[1], fieldBottom[j], 5,
                                                            r_correct_final[j])
    psnrTemp = test.psnr(template_img_pyramid[1][fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                         fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])],
                         warped_image[fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                         fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
    psnr.append(psnrTemp)
    cv2.imwrite(ROI_of_template, template_img_pyramid[1][fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                                 fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
    cv2.imwrite(ROI_of_target, warped_image[fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                               fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
    cv2.imwrite(name, warped_image)
    cv2.imwrite(name_of_difference, difference)
    cv2.imwrite(name_of_legend, legend)

print('PSNR:', psnr)

print('end')
timeGap = time.time() - recordTime
if timeGap >= 1:
    recordTime += timeGap
    showTime_String = test.get_lapseTime(startTime, time.strftime("%H%M%S"))
    print('run time :', showTime_String)
