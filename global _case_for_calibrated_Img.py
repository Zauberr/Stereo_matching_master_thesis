import time

import calibrated_img as cali

import numpy as np
import cv2

recordTime = time.time()
startTime = time.strftime("%H%M%S")

test = cali.calibrated_image('Photo/wide-left-rectified-8.jpg', 'Photo/wide-right-rectified-8.jpg')

# build the image Pyramid(pyrNum doesn't include original image)
# 0 layer is the original Image
pyrNum = 2
pyramid_2_transform, pyramid_of_form = test.build_image_pyramid(pyrNum)

# for calibrated_img H_{\infinity} is I and e is [[1],[0],[0]]

H_inf = np.eye(3)
e = np.array([[1], [0], [0]])


# show the image pyrmid
for i in range(pyrNum + 1):
    name_of_transform = "pyramid_of_transform_" + str(i) + "_layer.png"
    name_of_form = "pyramid_of_form_" + str(i) + "_layer.png"
    cv2.imwrite(name_of_form, pyramid_of_form[i])
    cv2.imwrite(name_of_transform, pyramid_2_transform[i])

# three target area in bottom(with small label) layer of pyramid for form image
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
field_0 = [56, 104, 135, 46]
field_1 = [325, 48, 320, 157]
field_2 = [613, 76, 158, 76]
field = [field_0, field_1, field_2]

# initial P for top layer
'''
I have two ways to calculate initial parameter p.
    (1)Estimate the initial offset by myself
    (2)Use function calculate parameter to calculate the initial offset
but it's a little slow for the second method. So I first use the method (2) by first running
then use the result directly for the rest running.
But the result seems not right. I have try to change the parameter in function, but it doesn't work.
In the end I have used the method (1).
p:[[0.19047970418103355, -0.03929862198602302, -16.167959950749662], 
[-0.7880955826338775, 0.0011705369114632688, 65.67203317623904], 
[-0.5944404867223292, -0.002228775452325685, 92.63052599763981]]
[[-0.22785316756177842, 0.0017344411639721067, -1.7686455373652272], 
[-0.716416132858829, -0.002629595379278235, 59.3717156238386], 
[-0.521593412357348, -0.016965646574616713, 81.08151325366093]]
'''

# change the field to top layer
for i in range(3):
    for j in range(4):
        field[i][j] = int(field[i][j] / 2 ** pyrNum)

a_0, b_0, c_0 = 0, 0, -10  # test.calulate_paramater(field[0])
a_1, b_1, c_1 = 0, 0, -10  # test.calulate_paramater(field[1])
a_2, b_2, c_2 = 0, 0, -10  # test.calulate_paramater(field[2])
q = [[a_0, b_0, c_0], [a_1, b_1, c_1], [a_2, b_2, c_2]]
# print(p)

# calculate q for different patches

for i in range(3):  # for different patch
    field_x = field[i]
    for j in range(pyrNum, -1, -1):  # for different layer

        image_warp = test.warp_image(q[i], e, H_inf, pyramid_2_transform[j], pyramid_of_form[j])
        name = "first_Warped_img_of_layer_" + str(j) + "_for_Patch" + str(i) + ".png"
        cv2.imwrite(name, image_warp)
        Delta_q = test.improve_parameter(pyramid_of_form[j], image_warp, field_x)

        count = 0
        while abs(Delta_q[0, 0]) > 0.001 or abs(Delta_q[1, 0]) > 0.001 or abs(Delta_q[2, 0]) > 0.001:
            count = count + 1
            if count > 50:
                # print('more than 50 cycles')
                break

            q[i][0] = q[i][0] + Delta_q[0, 0]
            q[i][1] = q[i][1] + Delta_q[1, 0]
            q[i][2] = q[i][2] + Delta_q[2, 0]

            image_warp = test.warp_image(q[i], e, H_inf, pyramid_2_transform[j], pyramid_of_form[j])
            Delta_q = test.improve_parameter(pyramid_of_form[j], image_warp, field_x)

        cycles = "cycles_of_layer_" + str(j) + "_for_" + str(i) + "_patch_:"
        parameter = "parameter_of_layer_" + str(j) + "_for_" + str(i) + "_patch_:"
        # print(cycles, count - 1)
        # print(parameter, q[i])

        difference = abs(
            image_warp[field_x[2]:(field_x[2] + field_x[3]), field_x[0]:(field_x[0] + field_x[1])] -
            pyramid_of_form[j][field_x[2]:(field_x[2] + field_x[3]), field_x[0]:(field_x[0] + field_x[1])])

        ROI_of_form = "ROI_of_form_of_" + str(j) + "_layer_for_" + str(i) + "_patch.png"
        ROI_of_trans = "ROI_of_transformed_img_of_" + str(j) + "_layer_for_" + str(i) + "_patch.png"
        defference = "defference_of_" + str(j) + "_layer_for_" + str(i) + "_patch.png"
        warped_img = "last_warped_img_of_" + str(j) + "_layer_for_" + str(i) + "_patch.png"

        # cv2.imwrite(ROI_of_form,
        #             pyramid_of_form[j][field_x[2]:(field_x[2] + field_x[3]), field_x[0]:(field_x[0] + field_x[1])])
        # cv2.imwrite(ROI_of_trans,
        #             image_warp[field_x[2]:(field_x[2] + field_x[3]), field_x[0]:(field_x[0] + field_x[1])])
        # cv2.imwrite(defference, difference)
        # cv2.imwrite(warped_img, image_warp)

        # change the parameter to  next layer
        if j != 0:
            q[i][2] = q[i][2] * 2
            for k in range(4):
                field_x[k] = field_x[k] * 2
print("q:", q)

'''
q: [[-0.021413044272087848, -0.005677684070379102, -34.76980894308102], 
[-0.016808314197713198, -0.0012181651474980541, -32.94670171500112], 
[0.03962871941675908, -0.003261624083959941, -60.3054005493301]]
'''

# calculate E and epsilon
count = 1
while count < 50:

    A_Matrix = np.zeros((12, 12))  # for dyadic product
    B_Matrix = np.zeros((12, 1))

    for i in range(3):  # i patch
        field_x = field[i]
        image_warp = test.warp_image(q[i], e, H_inf, pyramid_2_transform[0], pyramid_of_form[0])
        ImageDerivativeX = cv2.Sobel(pyramid_2_transform[0], -1, 1, 0, ksize=5)
        ImageDerivativeY = cv2.Sobel(pyramid_2_transform[0], -1, 0, 1, ksize=5)
        ImageDerivativeX_warp = test.warp_image(q[i], e, H_inf, ImageDerivativeX, pyramid_of_form[0])
        ImageDerivativeY_warp = test.warp_image(q[i], e, H_inf, ImageDerivativeY, pyramid_of_form[0])

        q_array = np.array(q[i]).reshape(1, 3)  # q_array means q_i^T

        for x in range(field_x[0], field_x[0] + field_x[1]):  # j pixel
            for y in range(field_x[2], field_x[2] + field_x[3]):
                # calculate D_H, D_e, D_N, H_i
                x_ij = np.array([[1], [2], [1]])  # x_ij in form img

                H = np.dot((H_inf + np.dot(e, q_array)), x_ij)

                D_N = np.array([[1, 0, 0], [0, 1, 0]]) * H[2] ** (-1) - \
                      H[2] ** (-2) * np.dot(H[0:2], np.array([[0, 0, 1]]))

                D_H = np.kron(np.transpose(x_ij), np.ones((3, 3)))

                D_e = np.dot(q_array, x_ij)

                # here calculate what I use in the big Matrix

                D_H_29 = np.dot(D_N, D_H)  # the number behind H means the dimension

                D_e_23 = D_e * D_N

                # calculate D_I
                """
                ？？？？
                I am not sure that we use warp_perspective functiong. I get a new image named warp_image.
                In this way I_2(H_i) is tha same as image_warp(x_ij[0:2])
                Dose it represent the the iterm I_2 in equation?
                I will use difference between I(x+1, y) and I(x-1, y) to replace differential DI_x(x,y)
                we only have the point at integer point. So we don't need to calculate integer smaller than 
                x.
                
                After discussion and process, I decide to first use sobel(I have search much, there are different mask, for 
                example Robert, Sobel, Prewitt)， Sobel is besser. So I use it temporarily.
                """

                D_I_12 = np.array([ImageDerivativeX_warp[y, x], ImageDerivativeY_warp[y, x]])

                # then calculate D_I*D_H and D_I*D_e

                M_1 = np.dot(D_I_12, D_H_29)
                M_2 = np.dot(D_I_12, D_e_23)

                M = np.concatenate((M_1, M_2), axis=0).reshape(1, 12)

                # I_1-I_2

                d = int(pyramid_of_form[0][y, x]) - int(image_warp[y, x])

                """
                I am not sure, if I can use the dyadic product here, too. I have google it, but not so clear. 
                I just use it here.
                """
                A_Matrix = A_Matrix + np.dot(np.transpose(M), M)
                B_Matrix = B_Matrix + d * np.transpose(M)

    # calculate Delta (from matrix to vector by column)
    Delta = np.dot(np.linalg.pinv(A_Matrix), B_Matrix)

    '''
    if we renormalize the Delta, when should we stop the loop?
    '''
    if np.linalg.norm(Delta, ord=2) < 0.0001:
        print('the error is smaller than 0.0001 ')
        break

    if np.linalg.norm(Delta, ord=2) < 0.1 or np.linalg.norm(Delta, ord=2) > 10:
        Delta = Delta / np.linalg.norm(Delta)

    Delta_H_inf = np.array([[Delta[0, 0], Delta[3, 0], Delta[6, 0]], [Delta[1, 0], Delta[4, 0], Delta[7, 0]],
                            [Delta[2, 0], Delta[5, 0], Delta[8, 0]]])
    Delta_e = np.array([[Delta[9, 0]], [Delta[10, 0]], [Delta[11, 0]]])

    H_inf = H_inf + Delta_H_inf

    e = e + Delta_e

    print('Loop Number:', count)
    count = count + 1

print('H_inf:', H_inf)
print('e:', e)

timeGap = time.time() - recordTime
if timeGap >= 1:  # 这是按1秒设置的，可以根据实际需要设置
    recordTime += timeGap
    showTime_String = test.get_lapseTime(startTime, time.strftime("%H%M%S"))
    print('run time :', showTime_String)
