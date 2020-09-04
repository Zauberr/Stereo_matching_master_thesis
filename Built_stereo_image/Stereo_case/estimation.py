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
text_name = folderName + '/Estimation.txt'
text = open(text_name, 'w+')

text_name2 = folderName + '/Estimation_number.txt'
text_num = open(text_name2, 'w+')

# three target area in bottom(with small label) layer of pyramid for left_img(template_img 1)
'''
Rectangle Selection
  X: 327
  Y: 237
  Width: 166
  Height: 149
Rectangle Selection
  X: 792
  Y: 255
  Width: 167
  Height: 164

Rectangle Selection
  X: 523
  Y: 517
  Width: 161
  Height: 80
In form [x, width, y, height]
'''
fieldNum = 3
field_0 = [327, 166, 237, 149]
field_1 = [792, 167, 255, 164]
field_2 = [523, 161, 517, 80]
fieldBottom = [field_0, field_1, field_2]

# threshold definition
qThreshold = 1e-4
heThreshold = 1e-5

# H transform template to target(left to right). Left is template image. Right is target image.
test = cali.calibrated_image('R.png', 'L.png', folderName, qThreshold, heThreshold, 7)

# build the image Pyramid(pyrNum doesn't include original image)
# 0 layer is the original Image
pyrNum = 2
target_img_pyramid, template_img_pyramid = test.build_image_pyramid(pyrNum, 0)

# initialization of H_inf, e and q_n from the GPS and INS data.
H_inf = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]], dtype=np.float32)
e = np.array([[-5.69822333e+02], [-4.57549368e-03], [1.52790722e-05]], dtype=np.float32)

# build the field to top layer
fieldTop = copy.deepcopy(fieldBottom)
for i in range(fieldNum):
    for j in range(4):
        fieldTop[i][j] = int(fieldTop[i][j] / 2 ** pyrNum)

a_0, b_0, c_0 = -5.041e-05, -2.539e-05, 1.220e-01
a_1, b_1, c_1 = 6.809e-05, -2.414e-05, 4.647e-02
a_2, b_2, c_2 = 0.000e+00, 1.098e-04, 3.289e-02
q = [np.array([a_0, b_0, c_0], dtype=np.float32).reshape(3, 1),
     np.array([a_1, b_1, c_1], dtype=np.float32).reshape(3, 1),
     np.array([a_2, b_2, c_2], dtype=np.float32).reshape(3, 1)]

a, b = 1.0, 0.0
r_correct = [np.array([a, b], dtype=np.float32).reshape(2, 1), np.array([a, b], dtype=np.float32).reshape(2, 1),
             np.array([a, b], dtype=np.float32).reshape(2, 1)]
# blurred image
q_final_blur, r_correct_final_blur, qConvergeSyb_blur = test.q_update(fieldBottom, pyrNum, fieldNum,
                                                                      target_img_pyramid[0],
                                                                      template_img_pyramid[0], q, e, H_inf, r_correct,
                                                                      0, 1)
# original image
q_final, r_correct_final, qConvergeSyb = test.q_update(fieldBottom, pyrNum, fieldNum, target_img_pyramid[1],
                                                       template_img_pyramid[1], q_final_blur, e, H_inf,
                                                       r_correct_final_blur, 0, 1)

psnr = []
H_inf_gt, e_gt, q_gt = test.read_ground_truth('ground_truth_number.txt')
for j in range(fieldNum):
    name = folderName + '/warped_image_of_' + 'patch_' + str(j + 1) + '.png'
    ROI_of_template = folderName + '/patch_' + str(j + 1) + '_ROI_of_template.png'
    ROI_of_target = folderName + '/patch_' + str(j + 1) + '_ROI_of_warped_image.png'
    name_of_difference = folderName + '/patch_' + str(j + 1) + '_difference.png'
    warped_image = test.warp_image(q_final[j], e, H_inf, target_img_pyramid[1],
                                   template_img_pyramid[1])
    warped_image_gt = test.warp_image(q_gt[j], e_gt, H_inf_gt, target_img_pyramid[1], template_img_pyramid[1])

    difference, legend = test.show_difference_of_warp_image(warped_image, warped_image_gt, fieldBottom[j], 5,
                                                            r_correct_final[j])
    psnrTemp = test.psnr(warped_image_gt[fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                         fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])],
                         warped_image[fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                         fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
    psnr.append(psnrTemp)
    cv2.imwrite(ROI_of_template, warped_image_gt[fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                                 fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
    cv2.imwrite(ROI_of_target, warped_image[fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                               fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
    cv2.imwrite(name, warped_image)
    cv2.imwrite(name_of_difference, difference)

print('Final Result:')

print('q:', q_final)
print('r_correct:', r_correct_final)
print('PSNR:', psnr)

print('q:', file=text)
print(q_final, file=text)

print('r_correct:', file=text)
print(r_correct_final, file=text)

print('PSNR:', file=text)
print(psnr, file=text)

# calculate RMSD and maximum displacement
RMSD, d_max, RMSD_each, d_max_each = test.calculate_RMSD(H_inf_gt, e_gt, q_gt, H_inf, e, q_final, fieldNum,
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

for i in range(0, 3):
    for j in range(0, 3):
        print(q_final[i][j, 0], file=text_num)

for i in range(0, 3):
    for j in range(0, 2):
        print(r_correct_final[i][j, 0], file=text_num)

text.close()
text_num.close()
timeGap = time.time() - recordTime
if timeGap >= 1:
    recordTime = timeGap
    showTime_String = test.get_lapseTime(startTime, time.strftime("%H%M%S"))
    print('run time :', showTime_String)
