import calibrated_img as cali
import time
import cv2
import numpy as np
import os
import copy

recordTime = time.time()
startTime = time.strftime("%H%M%S")

folderName = 'LocalResult' + '_rectified_Img_with_barn1'
if not os.path.exists(folderName):
    os.makedirs(folderName)

text_name = folderName + '/result.txt'
text = open(text_name, 'w+')

'''
Rectangle Selection 0
  X: 14
  Y: 206
  Width: 92
  Height: 145

Rectangle Selection 1
  X: 230
  Y: 279
  Width: 85
  Height: 84

Rectangle Selection 2
  X: 359
  Y: 15
  Width: 59
  Height: 101
  In form [x, width, y, height]
'''
fieldNum = 3
field_0 = [14, 92, 206, 145]
field_1 = [230, 85, 279, 84]
field_2 = [359, 59, 15, 101]
fieldBottom = [field_0, field_1, field_2]

qThreshold = 1e-3
heThreshold = 0
# first target second template, from template to target from right to left; 6 right to 2 left
test = cali.calibrated_image('/Users/zauber/Desktop/Stereo_matching/Photo/barn1/im2.ppm',
                             '/Users/zauber/Desktop/Stereo_matching/Photo/barn1/im6.ppm',
                             folderName, qThreshold, heThreshold, 3)

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

a_0, b_0, c_0 = 0, 0, 10
a_1, b_1, c_1 = 0, 0, 10
a_2, b_2, c_2 = 0, 0, 7
q = [np.array([a_0, b_0, c_0], dtype=np.float32).reshape(3, 1),
     np.array([a_1, b_1, c_1], dtype=np.float32).reshape(3, 1),
     np.array([a_2, b_2, c_2], dtype=np.float32).reshape(3, 1)]

a, b = 1.0, 0.0
# behind is all the same
r_correct = [np.array([a, b], dtype=np.float32).reshape(2, 1), np.array([a, b], dtype=np.float32).reshape(2, 1),
             np.array([a, b], dtype=np.float32).reshape(2, 1)]

q_final, r_correct_final, qConvergeSyb = test.q_update(fieldBottom, pyrNum, fieldNum, target_img_pyramid[0],
                                                       template_img_pyramid[0], q, e, H_inf, r_correct, 0, 0)

# compare with the ground truth disparity image
dispariy = test.disparity_of_stereo_img(q_final, fieldNum, fieldBottom)
ground_truth = cv2.imread('/Users/zauber/Desktop/Stereo_matching/Photo/barn1/disp6.pgm', cv2.IMREAD_GRAYSCALE)

print('q_final:', q_final)
print('r_correct:', r_correct_final)
print('q_final:', file=text)
print(q_final, file=text)
print('r_correct_final:', file=text)
print(r_correct_final, file=text)

psnr = []
for j in range(fieldNum):
    name = folderName + '/warped_image_of_' + 'patch_' + str(j + 1) + '.png'
    ROI_of_template = folderName + '/patch_' + str(j + 1) + '_ROI_of_template.png'
    ROI_of_target = folderName + '/patch_' + str(j + 1) + '_ROI_of_warped_image.png'
    name_of_difference = folderName + '/patch_' + str(j + 1) + '_difference.png'
    name_of_legend = folderName + '/patch_' + str(j + 1) + '_legend.png'

    warped_image = test.warp_image(q_final[j], e, H_inf, target_img_pyramid[0], template_img_pyramid[0])
    difference, legend = test.show_difference_of_warp_image(warped_image, template_img_pyramid[1], fieldBottom[j], 5,
                                                            r_correct_final[j])
    psnrTemp = test.psnr(template_img_pyramid[0][fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                         fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])],
                         warped_image[fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                         fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
    psnr.append(psnrTemp)

    cv2.imwrite(ROI_of_template, template_img_pyramid[0][fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                                 fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
    cv2.imwrite(ROI_of_target, warped_image[fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                               fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
    cv2.imwrite(name, warped_image)
    cv2.imwrite(name_of_difference, difference)
    cv2.imwrite(name_of_legend, legend)

    # evaluation
    name_of_ground_truth_disparity = folderName + '/patch_' + str(j + 1) + 'gt_disparity.png'
    name_of_disparity = folderName + '/patch_' + str(j + 1) + 'disparity.png'
    cv2.imwrite(name_of_ground_truth_disparity, ground_truth[fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                                                fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
    cv2.imwrite(name_of_disparity, dispariy[j])

print('PSNR:', psnr)
print('PSNR:', file=text)
print(psnr, file=text)

# RMSD
RMSD, d_max, RMSD_each, d_max_each = test.calculate_RMSD_stereo(ground_truth, H_inf, e, q_final, fieldNum, fieldBottom)

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
print('end')

timeGap = time.time() - recordTime
if timeGap >= 1:
    recordTime += timeGap
    showTime_String = test.get_lapseTime(startTime, time.strftime("%H%M%S"))
    print('run time :', showTime_String)
