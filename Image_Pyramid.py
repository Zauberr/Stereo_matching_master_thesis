import cv2
import time
import numpy as np
import calibrated_img as cali

recordTime = time.time()
startTime = time.strftime("%H%M%S")

test = cali.calibrated_image('Photo/wide-left-rectified-8.jpg', 'Photo/wide-right-rectified-8.jpg',
                             [[368, 41, 348, 92]])

# pyrNum doesn't include original image
pyrNum = 2
target_img_pyramid, template_img_pyramid = test.build_image_pyramid(pyrNum)

for i in range(pyrNum + 1):
    name_of_target = "target_ing_pyramid_of" + str(i) + "_layer.png"
    name_of_template = "template_img_pyramid_of" + str(i) + "_layer.png"
    cv2.imwrite(name_of_target, target_img_pyramid[i])
    cv2.imwrite(name_of_template, template_img_pyramid[i])

a, b, c = (0, 0, 40)
# or test.calulate_paramater([351, 76, 277, 222])
# this is the estimation of parameter of original image. It has to be changed
# to the parameter of the top layer in image pyramid

# a = a / (2 ** pyrNum)
# b = b / (2 ** pyrNum)
c = c / (2 ** pyrNum)

print('top layer a = ', a)
print('top layer b = ', b)
print('top layer c = ', c)

for i in range(pyrNum, -1, -1):

    x_min = int(311 / (2 ** i))
    x_width = int(74 / (2 ** i))
    y_min = int(327 / (2 ** i))
    y_height = int(145 / (2 ** i))

    a = 0
    b = 0
    # warped_img = test.warp_image((a, b, c), np.array([[1], [0], [0]]), np.eye(3), test.target_img, test.template_img)
    # name = "first_warped_img_of_layer_" + str(i) + ".png"
    # cv2.imwrite(name, warped_img)
    Delta_p = test.improve_parameter((a, b, c), np.array([[1], [0], [0]]), np.eye(3), target_img_pyramid[i],
                                     template_img_pyramid[i], [x_min, x_min, y_min, y_height])

    count = 0
    while abs(Delta_p[0, 0]) > 0.0001 or abs(Delta_p[1, 0]) > 0.0001 or abs(Delta_p[2, 0]) > 0.0001:
        count = count + 1
        if count > 200:
            print('more than 200 cycles')
            break

        a = a + Delta_p[0, 0]
        b = b + Delta_p[1, 0]
        c = c + Delta_p[2, 0]

        Delta_p = test.improve_parameter((a, b, c), np.array([[1], [0], [0]]), np.eye(3), target_img_pyramid[i],
                                         template_img_pyramid[i], [x_min, x_min, y_min, y_height])

    warped_img = test.warp_image((a, b, c), np.array([[1], [0], [0]]), np.eye(3), target_img_pyramid[i],
                                 template_img_pyramid[i])
    cycles = "cycles_of_layer_" + str(i) + ":"
    parameter = "parameter_of_layer_" + str(i) + ":"
    print(cycles, count - 1)
    print(parameter, a, b, c)

    difference = abs(
        warped_img[x_min:(y_min + y_height), x_min:(x_min + x_width)] - template_img_pyramid[i][
                                                                        x_min:(y_min + y_height),
                                                                        x_min:(x_min + x_width)])

    ROI_of_template = "ROI_of_template_of_" + str(i) + "_layer.png"
    ROI_of_target = "ROI_of_target_of_" + str(i) + "_layer.png"
    name_of_defference = "defference_of_" + str(i) + "_layer.png"
    name_of_warped_img = "last_warped_img_of_" + str(i) + "_layer.png"

    cv2.imwrite(ROI_of_template, template_img_pyramid[i][y_min:(y_min + y_height), x_min:(x_min + x_width)])
    cv2.imwrite(ROI_of_target, warped_img[y_min:(y_min + y_height), x_min:(x_min + x_width)])
    cv2.imwrite(name_of_defference, difference)
    cv2.imwrite(name_of_warped_img, warped_img)

    # change the parameter to  next layer
    if i != 0:
        c = c * 2

print('final parameter:', (a, b, c))
timeGap = time.time() - recordTime
if timeGap >= 1:
    recordTime += timeGap
    showTime_String = test.get_lapseTime(startTime, time.strftime("%H%M%S"))
    print('run time :', showTime_String)

'''
top layer a =  0
top layer b =  0
top layer c =  10.0
more than 200 cycles
cycles_of_layer_2: 200
parameter_of_layer_2: 0.027374759687591312 -0.0005090017957213536 7.457675111009494
more than 200 cycles
cycles_of_layer_1: 200
parameter_of_layer_1: 0.024897962521877696 -0.001957387199604366 15.695306211578858
more than 200 cycles
cycles_of_layer_0: 200
parameter_of_layer_0: 0.019787829360771877 0.000689775455659331 32.29676156898562
final parameter: (0.019787829360771877, 0.000689775455659331, 32.29676156898562)
run time : 00:04:35
'''
