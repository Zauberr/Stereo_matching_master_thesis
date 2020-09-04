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
  X: 239
  Y: 142
  Width: 190
  Height: 157
Rectangle Selection
  X: 792
  Y: 159
  Width: 222
  Height: 213
Rectangle Selection
  X: 535
  Y: 483
  Width: 246
  Height: 186
In form [x, width, y, height]
'''
fieldNum = 3
field_0 = [239, 190, 142, 157]
field_1 = [792, 222, 159, 213]
field_2 = [535, 246, 483, 186]
fieldBottom = [field_0, field_1, field_2]

# threshold definition
qThreshold = 1e-5
heThreshold = 1e-6

# left as target image, right as template image. H transform template to target(right to left)
test = cali.calibrated_image('R.png', 'L.png', folderName, qThreshold, heThreshold, 7)

# build the image Pyramid(pyrNum doesn't include original image)
# 0 layer is the original Image
pyrNum = 2
target_img_pyramid, template_img_pyramid = test.build_image_pyramid(pyrNum, 0)

# initialization of H_inf, e and q_n from the GPS and INS data.
H_inf = np.array([[9.67e-01, -3.13e-02, 1.42e+02],
                  [1.54e-02, 1.00e+00, -1.94e+01],
                  [-4.93e-05, 5.50e-06, 1.02e+00]], dtype=np.float32)
e = np.array([-1.19e+03, -1.55e+00, 4.98e-03], dtype=np.float32).reshape(3, 1)

# build the field to top layer
fieldTop = copy.deepcopy(fieldBottom)
for i in range(fieldNum):
    for j in range(4):
        fieldTop[i][j] = int(fieldTop[i][j] / 2 ** pyrNum)

a_0, b_0, c_0 = -4.45e-05, -2.22e-05, 1.08e-01
a_1, b_1, c_1 = 4.89e-05, -1.18e-05, 5.73e-02
a_2, b_2, c_2 = -1.00e-05, 1.09e-05, 9.83e-02
q = [np.array([a_0, b_0, c_0], dtype=np.float32).reshape(3, 1),
     np.array([a_1, b_1, c_1], dtype=np.float32).reshape(3, 1),
     np.array([a_2, b_2, c_2], dtype=np.float32).reshape(3, 1)]

a, b = 1.0, 0.0
r_correct = [np.array([a, b], dtype=np.float32).reshape(2, 1), np.array([a, b], dtype=np.float32).reshape(2, 1),
             np.array([a, b], dtype=np.float32).reshape(2, 1)]

max_loop_Number = 6

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
    for j in range(fieldNum):
        name = folderName + '/warped_image_of_' + 'patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '.png'
        ROI_of_template = folderName + '/patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '_ROI_of_template.png'
        ROI_of_target = folderName + '/patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '_ROI_of_warped_image.png'
        name_of_difference = folderName + '/patch_' + str(j + 1) + '_of_cycle_' + str(i + 1) + '_difference.png'

        warped_image = test.warp_image(qTemporary[j], e_temporary, H_inf_temporary, target_img_pyramid[1],
                                       template_img_pyramid[1])
        difference, legend = test.show_difference_of_warp_image(warped_image, template_img_pyramid[1], fieldBottom[j],
                                                                5, r_correctTemp[j])
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

psnr = []
H_inf_gt, e_gt, q_gt = test.read_ground_truth('ground_truth_number.txt')
for j in range(fieldNum):
    name = folderName + '/warped_image_of_' + 'patch_' + str(j + 1) + '.png'
    ROI_of_template = folderName + '/patch_' + str(j + 1) + '_ROI_of_template.png'
    ROI_of_target = folderName + '/patch_' + str(j + 1) + '_ROI_of_warped_image.png'
    name_of_difference = folderName + '/patch_' + str(j + 1) + '_difference.png'
    warped_image = test.warp_image(qTemporary[j], e_temporary, H_inf_temporary, target_img_pyramid[1],
                                   template_img_pyramid[1])
    warped_image_gt = test.warp_image(q_gt[j], e_gt, H_inf_gt, target_img_pyramid[1], template_img_pyramid[1])
    difference, legend = test.show_difference_of_warp_image(warped_image, warped_image_gt, fieldBottom[j], 5,
                                                            r_correctTemp[j])
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
print('H_inf:', H_inf_temporary)
print('e:', e_temporary)
print('q:', qTemporary)
print('r_correct:', r_correctTemp)
print('PSNR:', psnr)

print('H_inf:', file=text)
print(H_inf_temporary, file=text)

print('e:', file=text)
print(e_temporary, file=text)

print('q:', file=text)
print(qTemporary, file=text)

print('r_correct:', file=text)
print(r_correctTemp, file=text)

print('PSNR:', file=text)
print(psnr, file=text)

# calculate RMSD and maximum displacement
RMSD, d_max, RMSD_each = test.calculate_RMSD(H_inf_gt, e_gt, q_gt, H_inf_temporary, e_temporary, qTemporary, fieldNum,
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

for i in range(0, 3):
    for j in range(0, 3):
        print(H_inf_temporary[i, j], file=text_num)

for i in range(0, 3):
    print(e_temporary[i, 0], file=text_num)

for i in range(0, 3):
    for j in range(0, 3):
        print(qTemporary[i][j, 0], file=text_num)

for i in range(0, 3):
    for j in range(0, 2):
        print(r_correctTemp[i][j, 0], file=text_num)

text.close()
text_num.close()
timeGap = time.time() - recordTime
if timeGap >= 1:
    recordTime = timeGap
    showTime_String = test.get_lapseTime(startTime, time.strftime("%H%M%S"))
    print('run time :', showTime_String)
