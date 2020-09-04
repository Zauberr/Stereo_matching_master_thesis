import time

import calibrated_img as cali

import numpy as np
import cv2

import copy

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
In form [x, width, y, height]
'''
fieldNum = 3
field_0 = [56, 104, 135, 46]
field_1 = [325, 48, 320, 157]
field_2 = [613, 76, 158, 76]
fieldBottom = [field_0, field_1, field_2]

"""
build a boundary for the three field in target_img.
but for this, if we need a color boundary, we have to ues the color image.
until now we only implement the algorithm for the gray image. So now we first use a boundary in white.
Then for the next step, I will extend my algorithm to the color image and use a color boundary
"""
boundary_fieldBottom_2_trans = [[100, 70, 140, 30], [370, 30, 340, 100], [680, 50, 170, 50]]

test = cali.calibrated_image('Photo/wide-left-rectified-8.jpg', 'Photo/wide-right-rectified-8.jpg',
                             boundary_fieldBottom_2_trans)

# build the image Pyramid(pyrNum doesn't include original image)
# 0 layer is the original Image

pyrNum = 2
target_img_pyramid, template_img_pyramid = test.build_image_pyramid(pyrNum)

# for calibrated_img H_{\infinity} is I and e is [[1],[0],[0]]

"""
H_inf e and q are the parameter of dehomogenization. When we warped the 2_transform_image to form_image, we have to use inv(H). 
And I have consider it in function warp_image.
"""

H_inf = np.eye(3)
e = np.array([[1],
              [0],
              [0]])

# show the image pyrmid

# for i in range(pyrNum + 1):
#     name_of_transform = "pyramid_of_transform_" + str(i) + "_layer.png"
#     name_of_form = "template_img_pyramid_" + str(i) + "_layer.png"
#     cv2.imwrite(name_of_form, template_img_pyramid[i])
#     cv2.imwrite(name_of_transform, target_img_pyramid[i])

# build the field to top layer
fieldTop = copy.deepcopy(fieldBottom)
for i in range(fieldNum):
    for j in range(4):
        fieldTop[i][j] = int(fieldTop[i][j] / 2 ** pyrNum)

a_0, b_0, c_0 = 0, 0, 10  # test.calulate_paramater(field[0])
a_1, b_1, c_1 = 0, 0, 10  # test.calulate_paramater(field[1])
a_2, b_2, c_2 = 0, 0, 10  # test.calulate_paramater(field[2])
q = [[a_0, b_0, c_0], [a_1, b_1, c_1], [a_2, b_2, c_2]]  # for top layer, is smaller
# print(q)

max_loop_Number = 20
q_temporary = test.q_update(fieldTop, pyrNum, fieldNum, target_img_pyramid, template_img_pyramid, q, e, H_inf)

for i in range(max_loop_Number):

    print("loop Number:", i + 1)

    # calculate q for different patches

    '''
    the output of the function is q for the lay 0(original image).
    So it is bigger. and the input q is the q of the top layer, it is smaller    
    '''

    # calculate E and epsilon
    H_inf_temporary, e_temporary = test.H_inf_and_e_update(fieldNum, fieldBottom, q_temporary, e, H_inf,
                                                           target_img_pyramid, template_img_pyramid)

    # show result for every cycle

    for j in range(fieldNum):
        name = 'warped_image_of_' + 'cycle_' + str(i + 1) + '_patch_' + str(j + 1) + '.png'
        ROI_of_template = 'ROI_of_template_of_cycle_' + str(i + 1) + '_patch_' + str(j + 1) + '.png'
        ROI_of_target = 'ROI_of_warped_image_of_cycle_' + str(i + 1) + '_patch_' + str(j + 1) + '.png'

        warped_image = test.warp_image(q_temporary[j], e_temporary, H_inf_temporary, target_img_pyramid[0],
                                       template_img_pyramid[0])

        cv2.imwrite(ROI_of_template, template_img_pyramid[0][fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                                     fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
        cv2.imwrite(ROI_of_target, warped_image[fieldBottom[j][2]:(fieldBottom[j][2] + fieldBottom[j][3]),
                                   fieldBottom[j][0]:(fieldBottom[j][0] + fieldBottom[j][1])])
        cv2.imwrite(name, warped_image)

    # for j in range(fieldNum):
    #     q_temporary[j][2] = q_temporary[j][2] / 4

    # Delta_q = np.array(q_temporary) - np.array(q)
    Delta_H_inf = H_inf_temporary - H_inf
    Delta_e = e_temporary - e

    if np.linalg.norm(Delta_e, ord=2) <= 0.01 or np.linalg.norm(Delta_H_inf, ord=2) <= 0.01:
        print("the result converges")
        break

    if i == max_loop_Number - 1:
        print("Until the max loop, it doesn't converge")

    # q = copy.deepcopy(q_temporary)
    H_inf = H_inf_temporary
    e = e_temporary

# change the q to bottom layer bigger

# for j in range(fieldNum):
#     q_temporary[j][2] = q_temporary[j][2] * 4

'''
H_inf e and q are the parameter of dehomogenization. When we warped the 2_transform_image to form_image, we have to use inv(H)
'''

print('Final Result:')
print('H_inf:', H_inf_temporary)
print('e:', e_temporary)
# print('q:', q_temporary)

timeGap = time.time() - recordTime
if timeGap >= 1:
    recordTime += timeGap
    showTime_String = test.get_lapseTime(startTime, time.strftime("%H%M%S"))
    print('run time :', showTime_String)
