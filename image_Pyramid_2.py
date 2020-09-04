import cv2
import time
import numpy as np
import calibrated_img as cali

recordTime = time.time()
startTime = time.strftime("%H%M%S")

test = cali.calibrated_image('Photo/left_0.jpg', 'Photo/right_0.jpg')

# pyrNum doesn't include original image
pyrNum = 2
pyramid_2_transform, pyramid_of_form = test.build_image_pyramid(pyrNum)

for i in range(pyrNum + 1):
    name_of_transform = "pyramid_of_transform_" + str(i) + "_layer.png"
    name_of_form = "pyramid_of_form_" + str(i) + "_layer.png"
    cv2.imwrite(name_of_form, pyramid_of_form[i])
    cv2.imwrite(name_of_transform, pyramid_2_transform[i])

a, b, c = (0, 0, -30)
# or test.calulate_paramater([351, 76, 277, 222])
# this is the estimation of parameter of original image. It has to be changed
# to the parameter of the top layer in image pyramid

a = a / (2 ** pyrNum)
b = b / (2 ** pyrNum)
c = c / (2 ** pyrNum)

print('top layer a = ', a)
print('top layer b = ', b)
print('top layer c = ', c)

for i in range(pyrNum, -1, -1):

    x_min = int(734 / (2 ** i))
    x_width = int(504 / (2 ** i))
    y_min = int(848 / (2 ** i))
    y_height = int(138 / (2 ** i))

    image_warp = test.warp_image((a, b, c), pyramid_2_transform[i], pyramid_of_form[i])
    name = "first_warped_img_of_layer_" + str(i) + ".png"
    cv2.imwrite(name, image_warp)
    Delta_p = test.improve_parameter(pyramid_of_form[i], image_warp, [x_min, x_width, y_min, y_height])

    count = 0
    while abs(Delta_p[0, 0]) > 0.001 or abs(Delta_p[1, 0]) > 0.001 or abs(Delta_p[2, 0]) > 0.001:
        count = count + 1
        if count > 100:
            print('more than 100 cycles')
            break

        a = a + 0.1 * Delta_p[0, 0]
        b = b + 0.1 * Delta_p[1, 0]
        c = c + 0.1 * Delta_p[2, 0]

        image_warp = test.warp_image((a, b, c), pyramid_2_transform[i], pyramid_of_form[i])
        Delta_p = test.improve_parameter(pyramid_of_form[i], image_warp, [x_min, x_width, y_min, y_height])

    cycles = "cycles_of_layer_" + str(i) + ":"
    parameter = "parameter_of_layer_" + str(i) + ":"
    print(cycles, count - 1)
    print(parameter, a, b, c)

    difference = abs(
        image_warp[x_min:(y_min + y_height), x_min:(x_min + x_width)] - pyramid_of_form[i][x_min:(y_min + y_height),
                                                                        x_min:(x_min + x_width)])

    ROI_of_form = "ROI_of_form_of_" + str(i) + "_layer.png"
    ROI_of_trans = "ROI_of_transformed_img_of_" + str(i) + "_layer.png"
    defference = "defference_of_" + str(i) + "_layer.png"
    warped_img = "last_warped_img_of_" + str(i) + "_layer.png"

    cv2.imwrite(ROI_of_form, pyramid_of_form[i][y_min:(y_min + y_height), x_min:(x_min + x_width)])
    cv2.imwrite(ROI_of_trans, image_warp[y_min:(y_min + y_height), x_min:(x_min + x_width)])
    cv2.imwrite(defference, difference)
    cv2.imwrite(warped_img, image_warp)

    # change the parameter to  next layer
    if i != 0:
        a = a * 2
        b = b * 2
        c = c * 2

timeGap = time.time() - recordTime
if timeGap >= 1:  # 这是按1秒设置的，可以根据实际需要设置
    recordTime += timeGap
    showTime_String = test.get_lapseTime(startTime, time.strftime("%H%M%S"))
    print('run time :', showTime_String)
