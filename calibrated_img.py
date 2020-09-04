import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import copy
import math
import os


class calibrated_image(object):

    def __init__(self, target_img, template_img, folderName, qThreshold, heThreshold, Gaussian_value):
        # H transform from template to target_img

        temp_target_img = cv2.imread(target_img, cv2.IMREAD_GRAYSCALE)
        temp_template_img = cv2.imread(template_img, cv2.IMREAD_GRAYSCALE)

        self.target_img = np.float32(temp_target_img)
        self.template_img = np.float32(temp_template_img)
        # blur the image
        self.target_img_blur = cv2.GaussianBlur(self.target_img, (Gaussian_value, Gaussian_value), 10)
        self.template_img_blur = cv2.GaussianBlur(self.template_img, (Gaussian_value, Gaussian_value), 10)

        self.qThreshold = qThreshold
        self.heThreshold = heThreshold

        cv2.imwrite(folderName + '/original_template_img.png', self.template_img)
        cv2.imwrite(folderName + '/original_target_img.png', self.target_img)

    def q_update(self, field, pyrNum, fieldNum, target_img_pyramid, template_img_pyramid,
                 q, e, H_inf, r_correct, pyramidOrNot, radiometricOrNot):
        # I define q is a 3 * 1 dimension matrix. But it is in fact q_transpose. In this way, it's easy to multiple
        # e and q.T.
        convergeOrNot = 1
        if pyramidOrNot == 0:
            if radiometricOrNot == 1:
                q_in_func = copy.deepcopy(q)
                r_in_func = copy.deepcopy(r_correct)
                for i in range(fieldNum):  # for different patch

                    field_x = copy.deepcopy(field[i])

                    Delta = self.improve_parameter(q_in_func[i], e, H_inf, r_in_func[i], target_img_pyramid,
                                                   template_img_pyramid, radiometricOrNot, field_x)

                    count = 0
                    while np.linalg.norm(Delta) / np.linalg.norm(np.concatenate((q_in_func[i], r_in_func[i]), axis=0)) \
                            > self.qThreshold:
                        count = count + 1
                        if count > 50:
                            convergeOrNot = 0
                            print(i, 'patch: q cycle is more than 50 cycles')
                            break
                        Delta = Delta
                        q_in_func[i] = q_in_func[i] + Delta[0:3, 0].reshape(3, 1)
                        r_in_func[i] = r_in_func[i] + Delta[3:5, 0].reshape(2, 1)

                        Delta = self.improve_parameter(q_in_func[i], e, H_inf, r_in_func[i], target_img_pyramid,
                                                       template_img_pyramid, radiometricOrNot, field_x)
            else:
                q_in_func = copy.deepcopy(q)
                r_in_func = copy.deepcopy(r_correct)
                for i in range(fieldNum):  # for different patch

                    field_x = copy.deepcopy(field[i])

                    Delta = self.improve_parameter(q_in_func[i], e, H_inf, r_in_func[i], target_img_pyramid,
                                                   template_img_pyramid, radiometricOrNot, field_x)

                    count = 0
                    while np.linalg.norm(Delta) / np.linalg.norm(
                            q_in_func[i]) > self.qThreshold:  # Need to be adjusted
                        count = count + 1
                        if count > 50:
                            convergeOrNot = 0
                            print(i, 'patch: q cycle is more than 50 cycles')
                            break
                        Delta = Delta
                        q_in_func[i] = q_in_func[i] + Delta

                        Delta = self.improve_parameter(q_in_func[i], e, H_inf, r_in_func[i], target_img_pyramid,
                                                       template_img_pyramid, radiometricOrNot, field_x)
        else:

            q_in_func = copy.deepcopy(q)

            for i in range(fieldNum):  # for different patch

                field_x = copy.deepcopy(field[i])

                for j in range(pyrNum, -1, -1):  # for different layer

                    Delta_q = self.improve_parameter(q_in_func[i], e, H_inf, target_img_pyramid[j],
                                                     template_img_pyramid[j], field_x)

                    count = 0
                    while abs(Delta_q[0, 0]) > 0.01 or abs(Delta_q[1, 0]) > 0.01 or abs(
                            Delta_q[2, 0]) > 0.1:  # Need to be adjusted
                        count = count + 1
                        if count > 50:
                            print('q cycle is more than 50 cycles')
                            convergeOrNot = 0
                            break

                        Delta_q = Delta_q
                        q_in_func[i] = q_in_func[i] + Delta_q[i]

                        # warped_img = self.warp_image(q_in_func[i], e, H_inf, target_img_pyramid[j], template_img_pyramid[j])
                        Delta_q = self.improve_parameter(q_in_func[i], e, H_inf, target_img_pyramid[j],
                                                         template_img_pyramid[j], field_x)

                    # change the parameter to  next layer
                    if j != 0:
                        q_in_func[i][0, 2] = q_in_func[i][0, 2] * 2
                        for k in range(4):
                            field_x[k] = field_x[k] * 2

        print("q:", q_in_func)
        print('r_correct:', r_in_func)
        return q_in_func, r_in_func, convergeOrNot

    def H_inf_and_e_update(self, fieldNum, field, q, e, H_inf, r_correct, target_img_pyramid, template_img_pyramid):
        count = 0
        convergeOrNot = 0
        while count < 30:
            if count == 29:
                print('H and e cycle is more than 20 cycles')
            A_Matrix = np.zeros((12, 12))  # for dyadic product
            B_Matrix = np.zeros((12, 1))

            for i in range(fieldNum):  # i patch
                field_x = field[i]

                warped_img = self.warp_image(q[i], e, H_inf, target_img_pyramid, template_img_pyramid)
                ImageDerivativeX = cv2.Sobel(target_img_pyramid, -1, 1, 0, ksize=-1)
                ImageDerivativeY = cv2.Sobel(target_img_pyramid, -1, 0, 1, ksize=-1)
                ImageDerivativeX_warp = self.warp_image(q[i], e, H_inf, ImageDerivativeX, template_img_pyramid)
                ImageDerivativeY_warp = self.warp_image(q[i], e, H_inf, ImageDerivativeY, template_img_pyramid)

                q_array = q[i]

                for x in range(field_x[0], field_x[0] + field_x[1]):  # j pixel
                    for y in range(field_x[2], field_x[2] + field_x[3]):
                        # calculate D_H, D_e, D_N, H_i
                        x_ij = np.array([[x],
                                         [y],
                                         [1]])  # x_ij in form img

                        # dimension fo H is 3*3
                        H = np.matmul((H_inf + np.matmul(e, q_array.T)), x_ij)

                        # dimension of D_N is 2*3
                        D_N = np.array([[1, 0, 0],
                                        [0, 1, 0]]) / H[2] \
                              - np.matmul(H[0:2], np.array([[0, 0, 1]])) / (H[2] * H[2])

                        D_H = np.kron(np.transpose(x_ij), np.eye(3))

                        D_e = np.matmul(q_array.T, x_ij)

                        # here calculate what I use in the big Matrix

                        D_H_29 = np.matmul(D_N, D_H)  # the number behind H means the dimension

                        D_e_23 = D_e * D_N

                        # calculate D_I

                        D_I_12 = np.array([ImageDerivativeX_warp[y, x], ImageDerivativeY_warp[y, x]]).reshape(1, 2)

                        # then calculate D_I*D_H and D_I*D_e

                        M_1 = np.matmul(D_I_12, D_H_29)
                        M_2 = np.matmul(D_I_12, D_e_23)

                        # M is 1 * 12
                        M = np.concatenate((M_1, M_2), axis=1)

                        # difference between template image and warped image
                        d = r_correct[i][0, 0] * int(template_img_pyramid[y, x]) + r_correct[i][1, 0] - int(
                            warped_img[y, x])

                        # A_Matrix is 12 * 12
                        # B_Matrix is 12 * 1
                        A_Matrix = A_Matrix + np.matmul(np.transpose(M), M)
                        B_Matrix = B_Matrix + d * np.transpose(M)

            # calculate Delta (from matrix to vector by column) 14 * 1
            Delta = np.dot(np.linalg.pinv(A_Matrix), B_Matrix)

            '''
            if we renormalize the Delta, when should we stop the loop?
            '''
            H_inf_vec = np.array(H_inf).reshape(9, -1)

            vecHe = np.concatenate((H_inf_vec, e), axis=0)

            if np.linalg.norm(Delta) / np.linalg.norm(vecHe) < self.heThreshold:
                print('result of H_inf and e update converges')
                convergeOrNot = 1
                break

            Delta = Delta  # need to be adjusted

            Delta_H_inf = np.array([[Delta[0, 0], Delta[3, 0], Delta[6, 0]],
                                    [Delta[1, 0], Delta[4, 0], Delta[7, 0]],
                                    [Delta[2, 0], Delta[5, 0], Delta[8, 0]]])
            Delta_e = np.array([[Delta[9, 0]],
                                [Delta[10, 0]],
                                [Delta[11, 0]]])

            H_inf = H_inf + Delta_H_inf
            e = e + Delta_e
            count = count + 1
            '''
            theNorm = np.linalg.norm(H_inf) + np.linalg.norm(e)
            if theNorm < 0.1 or theNorm > 10:
                H_inf = H_inf / theNorm
                e = e / theNorm
                print('renormalize the H_inf and e')
            count = count + 1
            '''
        print('H:', H_inf)
        print('e:', e)
        return H_inf, e, convergeOrNot

    def improve_parameter(self, q, e, H_inf, r_correct, target_img, template_img, radiometricOrNot,
                          *correspondings_field):
        if radiometricOrNot == 1:
            A_matrix = np.zeros((5, 5))
            B_matrix = np.zeros((5, 1))

            warped_img = self.warp_image(q, e, H_inf, target_img, template_img)

            ImageDerivativeX = cv2.Sobel(target_img, -1, 1, 0, ksize=-1)
            ImageDerivativeX_warp = self.warp_image(q, e, H_inf, ImageDerivativeX, template_img)
            ImageDerivativeY = cv2.Sobel(target_img, -1, 0, 1, ksize=-1)
            ImageDerivativeY_warp = self.warp_image(q, e, H_inf, ImageDerivativeY, template_img)

            for field in correspondings_field:
                x_min_improve = field[0]
                x_max_improve = field[0] + field[1]
                y_min_improve = field[2]
                y_max_improve = field[2] + field[3]
                for y in range(y_min_improve, y_max_improve, 1):
                    for x in range(x_min_improve, x_max_improve, 1):
                        # coordinate in template img
                        x_ij = np.array([[x],
                                         [y],
                                         [1]])

                        # dimension fo H is 3*3
                        H = np.matmul((H_inf + np.matmul(e, q.T)), x_ij)

                        # dimension of D_N is 2*3
                        D_N = np.array([[1, 0, 0],
                                        [0, 1, 0]]) / H[2] \
                              - np.matmul(H[0:2], np.array([[0, 0, 1]])) / (H[2] * H[2])

                        # Derivative of I_2 (warped img)
                        D_I_12 = np.array([ImageDerivativeX_warp[y, x], ImageDerivativeY_warp[y, x]]).reshape(1, 2)

                        # D_I after dehomogenous 1*3
                        D_I_dehomogenous = np.matmul(D_I_12, D_N)

                        # M_1 is the part for Delta p in A_matrix. Dimension is 1 * 3
                        M_1 = np.matmul(np.matmul(D_I_dehomogenous, e), np.transpose(x_ij))

                        # M_2 is the part for Delta a and Delta b in A_matrix. Dimension is 1 * 2
                        M_2 = np.array([[template_img[y, x], 1]])

                        # M is the factor of variable dimension is 1*5
                        M = np.concatenate((M_1, -M_2), axis=1)

                        # difference between template image and warped image
                        d = r_correct[0, 0] * int(template_img[y, x]) + r_correct[1, 0] - int(warped_img[y, x])

                        # dimension of A_matrix is 5 * 5
                        # dimension of B_matrix is 5 * 1
                        A_matrix = A_matrix + np.matmul(np.transpose(M), M)
                        B_matrix = B_matrix + d * np.transpose(M)

            # dimension of q is 5 * 1
            Delta = np.matmul(np.linalg.pinv(A_matrix), B_matrix)
        else:
            A_matrix = np.zeros((3, 3))
            B_matrix = np.zeros((3, 1))

            warped_img = self.warp_image(q, e, H_inf, target_img, template_img)

            ImageDerivativeX = cv2.Sobel(target_img, -1, 1, 0, ksize=-1)
            ImageDerivativeX_warp = self.warp_image(q, e, H_inf, ImageDerivativeX, template_img)
            ImageDerivativeY = cv2.Sobel(target_img, -1, 0, 1, ksize=-1)
            ImageDerivativeY_warp = self.warp_image(q, e, H_inf, ImageDerivativeY, template_img)

            for field in correspondings_field:
                x_min_improve = field[0]
                x_max_improve = field[0] + field[1]
                y_min_improve = field[2]
                y_max_improve = field[2] + field[3]
                for y in range(y_min_improve, y_max_improve, 1):
                    for x in range(x_min_improve, x_max_improve, 1):
                        # coordinate in template img
                        x_ij = np.array([[x],
                                         [y],
                                         [1]])

                        # dimension fo H is 3*3
                        H = np.matmul((H_inf + np.matmul(e, q.T)), x_ij)

                        # dimension of D_N is 2*3
                        D_N = np.array([[1, 0, 0],
                                        [0, 1, 0]]) / H[2] \
                              - np.matmul(H[0:2], np.array([[0, 0, 1]])) / (H[2] * H[2])

                        # Derivative of I_2 (warped img)
                        D_I_12 = np.array([ImageDerivativeX_warp[y, x], ImageDerivativeY_warp[y, x]]).reshape(1, 2)

                        # D_I after dehomogenous 1*3
                        D_I_dehomogenous = np.matmul(D_I_12, D_N)

                        # M_1 is the part for Delta p in A_matrix. Dimension is 1 * 3
                        M_1 = np.matmul(np.matmul(D_I_dehomogenous, e), np.transpose(x_ij))

                        # M_2 is the part for Delta a and Delta b in A_matrix. Dimension is 1 * 2
                        # M_2 = np.array([[template_img[y, x], 1]])

                        # M is the factor of variable dimension is 1*3
                        M = M_1

                        # difference between template image and warped image
                        d = r_correct[0, 0] * int(template_img[y, x]) + r_correct[1, 0] - int(warped_img[y, x])

                        # dimension of A_matrix is 3 * 3
                        # dimension of B_matrix is 3 * 1
                        A_matrix = A_matrix + np.matmul(np.transpose(M), M)
                        B_matrix = B_matrix + d * np.transpose(M)

                        # dimension of q is 3 * 1
            Delta = np.matmul(np.linalg.pinv(A_matrix), B_matrix)

        return Delta

    def warp_image(self, q, e, H_inf, target_img, template_img):
        """
        Here we calculate the H from template to target, so when we warp image, we have to use inv(H)
        """
        H = H_inf + np.matmul(e, q.T)
        # print('p=', p)
        # print('e1=', e1)
        # print('H=', H)

        width_x = np.array(template_img).shape[1]
        height_y = np.array(template_img).shape[0]
        warped_img = cv2.warpPerspective(target_img, H, (width_x, height_y),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        # target_img_2 = cv2.warpPerspective(warped_img, np.linalg.inv(H), (width_x, height_y), flags=cv2.INTER_LINEAR)
        # print('size of left image', np.array(self.target_img).sh emphasizeape)
        # print('size of warped image :', np.array(warped_img).shape)
        # print('size of original image :', np.arraywujie(self.template_img).shape)
        return warped_img

    def show_difference_of_warp_image(self, warped_img, template_img, corresponding_field, factor, r):
        difference = warped_img - template_img
        difference_img = np.zeros((corresponding_field[3], corresponding_field[1], 3), dtype=np.float32)
        legend_img = np.zeros((40, 510, 3), dtype=np.float32)

        for x in range(0, 255):  # pixel in y row blue
            legend_img[0:40, x, 0] = 255 - x
        for x in range(255, 510):
            legend_img[0:40, x, 2] = x - 255

        for x in range(corresponding_field[0], corresponding_field[0] + corresponding_field[1]):  # j pixel
            for y in range(corresponding_field[2], corresponding_field[2] + corresponding_field[3]):  # i pixel
                if difference[y, x] > 0:  # blue
                    difference_img[y - corresponding_field[2], x - corresponding_field[0], 0] = difference[
                                                                                                    y, x] * 51
                    difference_img[y - corresponding_field[2], x - corresponding_field[0], 1] = 0
                    difference_img[y - corresponding_field[2], x - corresponding_field[0], 2] = 0
                elif difference[y, x] < 0:  # red
                    difference_img[y - corresponding_field[2], x - corresponding_field[0], 0] = 0
                    difference_img[y - corresponding_field[2], x - corresponding_field[0], 1] = 0
                    difference_img[y - corresponding_field[2], x - corresponding_field[0], 2] = (-difference[
                        y, x]) * 51
                else:
                    difference_img[y - corresponding_field[2], x - corresponding_field[0], :] = 0

        return difference_img, legend_img

    def psnr(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    def build_image_pyramid(self, NumLayer, use_or_not):
        if use_or_not == 1:
            layer_2_trans = np.copy(self.target_img)
            layer_of_form = np.copy(self.template_img)
            target_img_pyramid = [layer_2_trans]
            template_img_pyramid = [layer_of_form]

            for i in range(NumLayer):
                # NumLayer doesn't include the original image
                layer_2_trans = cv2.pyrDown(layer_2_trans)
                layer_of_form = cv2.pyrDown(layer_of_form)
                target_img_pyramid.append(layer_2_trans)
                template_img_pyramid.append(layer_of_form)
        else:
            target_img_pyramid = [self.target_img_blur, self.target_img]
            template_img_pyramid = [self.template_img_blur, self.template_img]
        return target_img_pyramid, template_img_pyramid

    def read_ground_truth(self, filename):
        f = open(filename, 'r')
        line = f.readlines()
        # print(line)
        H_inf_list = []
        e_list = []
        q_list = []

        for row in range(0, 3):
            H_inf_row_list = []
            for col in range(0, 3):
                data = line[3 * row + col]
                data = data[:-1]
                ele = float(data)
                H_inf_row_list.append(ele)
            H_inf_list.append(H_inf_row_list)

        for row in range(9, 12):
            data = line[row]
            data = data[:-1]
            ele = float(data)
            e_list.append(ele)

        for row in range(4, 7):
            q_row_list = []
            for col in range(0, 3):
                data = line[3 * row + col]
                data = data[:-1]
                ele = float(data)
                q_row_list.append(ele)
            q_row_list_array = np.array(q_row_list).reshape(3, 1)
            q_list.append(q_row_list_array)

        e = np.array(e_list).reshape(3, 1)
        H_inf = np.array(H_inf_list).reshape(3, 3)
        return H_inf, e, q_list

    def read_estimation_result(self, filename):
        f = open(filename, 'r')
        line = f.readlines()
        # print(line)
        H_inf_list = []
        e_list = []
        q_list = []
        r_list = []

        for row in range(0, 3):
            H_inf_row_list = []
            for col in range(0, 3):
                data = line[3 * row + col]
                data = data[:-1]
                ele = float(data)
                H_inf_row_list.append(ele)
            H_inf_list.append(H_inf_row_list)

        for row in range(9, 12):
            data = line[row]
            data = data[:-1]
            ele = float(data)
            e_list.append(ele)

        for row in range(4, 7):
            q_row_list = []
            for col in range(0, 3):
                data = line[3 * row + col]
                data = data[:-1]
                ele = float(data)
                q_row_list.append(ele)
            q_row_list_array = np.array(q_row_list).reshape(3, 1)
            q_list.append(q_row_list_array)

        for row in range(0, 2):
            r_row_list = []
            for col in range(0, 2):
                data = line[3 * 7 + 2 * row + col]
                data = data[:-1]
                ele = float(data)
                r_row_list.append(ele)
            r_row_list_array = np.array(r_row_list).reshape(2, 1)
            r_list.append(r_row_list_array)

        e = np.array(e_list).reshape(3, 1)
        H_inf = np.array(H_inf_list).reshape(3, 3)
        return H_inf, e, q_list, r_list

    def calculate_RMSD(self, H_inf_gt, e_gt, q_gt, H_inf, e, q, fieldNum, field):
        sum = 0.0
        RMSD_each = []
        D = 0.0
        n = 0.0
        d_max = 0.0
        d_max_each_list = []

        for i in range(fieldNum):  # i patch
            field_x = field[i]
            H_gt = H_inf_gt + e_gt @ q_gt[i].T
            H = H_inf + e @ q[i].T
            D_each = 0.0
            n_each = 0.0
            d_max_each = 0.0

            for x in range(field_x[0], field_x[0] + field_x[1]):  # j pixel column
                for y in range(field_x[2], field_x[2] + field_x[3]):  # i pixel row
                    x_ij = np.array([[x],
                                     [y],
                                     [1]], dtype=float)

                    y_ij_gt_3D = H_gt @ x_ij
                    y_ij_gt_2D = np.array([[y_ij_gt_3D[0, 0] / y_ij_gt_3D[2, 0]],
                                           [y_ij_gt_3D[1, 0] / y_ij_gt_3D[2, 0]]], dtype=float)

                    y_ij_3D = H @ x_ij
                    y_ij_2D = np.array([[y_ij_3D[0, 0] / y_ij_3D[2, 0]],
                                        [y_ij_3D[1, 0] / y_ij_3D[2, 0]]], dtype=float)

                    d = np.linalg.norm((y_ij_gt_2D - y_ij_2D))

                    if d > d_max_each:
                        d_max_each = d

                    D = D + d ** 2
                    D_each = D_each + d ** 2

                    n_each = n_each + 1
                    n = n + 1
            d_max_each_list.append(d_max_each)
            if d_max_each > d_max:
                d_max = d_max_each
            RMSD_each_part = math.sqrt(D_each / n_each)
            # print(RMSD_each_part)
            RMSD_each.append(RMSD_each_part)
        # print('D:', D)
        # print('n:', n)
        RMSD = math.sqrt(D / n)
        return RMSD, d_max, RMSD_each, d_max_each_list

    def disparity_of_stereo_img(self, q, field_num, field):
        disparity_list = []
        for i in range(field_num):
            field_x = field[i]
            q_x = q[i]
            disparity = np.zeros((field_x[3], field_x[1]))
            for x in range(field_x[0], field_x[0] + field_x[1]):  # j pixel
                for y in range(field_x[2], field_x[2] + field_x[3]):  # i pixel
                    disparity[y - field_x[2], x - field_x[0]] = ((1 + q_x[0, 0]) * x + q_x[1, 0] * y + q_x[
                        2, 0] - x) * 8

            disparity_list.append(disparity)
        return disparity_list

    def calculate_RMSD_stereo(self, disparity, H_inf, e, q, fieldNum, field):
        sum = 0.0
        RMSD_each = []
        D = 0.0
        n = 0.0
        d_max = 0.0
        d_max_each_list = []

        for i in range(fieldNum):  # i patch
            field_x = field[i]
            H = H_inf + e @ q[i].T
            D_each = 0.0
            n_each = 0.0
            d_max_each = 0.0
            disparity_x = disparity
            for x in range(field_x[0], field_x[0] + field_x[1]):  # j pixel column
                for y in range(field_x[2], field_x[2] + field_x[3]):  # i pixel row
                    x_ij = np.array([[x],
                                     [y],
                                     [1]], dtype=float)

                    y_ij_gt_2D = np.array([[(x + disparity_x[y, x] / 8)],
                                           [y]], dtype=float)

                    y_ij_3D = H @ x_ij
                    y_ij_2D = np.array([[y_ij_3D[0, 0] / y_ij_3D[2, 0]], [y_ij_3D[1, 0] / y_ij_3D[2, 0]]], dtype=float)

                    d = np.linalg.norm((y_ij_gt_2D - y_ij_2D))

                    if d > d_max_each:
                        d_max_each = d

                    D = D + d ** 2
                    D_each = D_each + d ** 2

                    n_each = n_each + 1
                    n = n + 1
            d_max_each_list.append(d_max_each)
            if d_max_each > d_max:
                d_max = d_max_each
            RMSD_each_part = math.sqrt(D_each / n_each)
            # print(RMSD_each_part)
            RMSD_each.append(RMSD_each_part)
        # print('D:', D)
        # print('n:', n)
        RMSD = math.sqrt(D / n)
        return RMSD, d_max, RMSD_each, d_max_each_list

    def calulate_paramater(self, field):
        # in field [x_min, width, y_min, height] in form image

        corr = np.zeros((100,))
        x_r_used = []
        x_and_y_l = []
        # in matrix y means rows, x means column. don't mix it

        # definition the corresponding region
        x_min = field[0]
        x_max = field[0] + field[1]
        y_min = field[2]
        y_max = field[2] + field[3]

        n = 0  # cout
        # for x_l in range(x_min, (x_min + 100), 1):  # x_l in area [x_r-10, x_r+10] to find
        #
        #     corr[n] = np.corrcoef(self.template_img[y_min:y_max, x_min:x_max][:],
        #                           self.target_img[y_min:y_max, x_l:(x_l + x_max - x_min)][:])[0, 1]
        #     n = n + 1

        for y_r in range(y_min, y_max, 1):
            for x_r in range(x_min, x_max, 1):
                # for the plane area in every column
                n = 0  # cout
                for x_l in range(x_r, (x_r + 20), 1):
                    # x_l in area [x_r-10, x_r+10] to find must changed for general case
                    corr[n] = np.corrcoef(self.template_img[y_r, (x_r - 5):(x_r + 6)],
                                          self.target_img[y_r, (x_l - 5):(x_l + 6)])[0, 1]
                    n = n + 1
                ind = (np.argmax(corr))
                x_and_y_l.append([y_r, (ind + x_r)])
                x_r_used.append(x_r)

        x_r_array = np.array(x_r_used)
        x_and_y_array = np.array(x_and_y_l)

        model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=1, normalize=False)
        model.fit(x_and_y_array, x_r_array)
        c = model.intercept_
        b, a = model.coef_
        a = a - 1
        return a, b, c

    def find_exact_field(self, small_and_exact_field, threshold):

        # Alpha in right image

        parameter = self.calulate_paramater(small_and_exact_field)
        warped_img = self.warp_image_L2R(parameter)
        D_matrix_from_left = abs(warped_img - self.template_img)
        # cv2.imwrite("warped_img.png", warped_img)
        # cv2.imwrite("image_right.png", self.template_img)

        for x in range(0, np.array(self.template_img).shape[1]):
            for y in range(0, np.array(self.target_img).shape[0]):
                if D_matrix_from_left[y, x] <= threshold:
                    self.gray_and_a_chanel_template_img[y, x, 1] = 1

        # Alpha in left image
        p = np.array([round(parameter[0], 3), round(parameter[1], 3),
                      round(parameter[2], 3)]).reshape(1, 3)
        e1 = np.array([1, 0, 0]).reshape(3, 1)
        I = np.eye(3)
        H = I + np.dot(e1, p)

        for x in range(0, np.array(self.target_img).shape[1]):
            for y in range(0, np.array(self.target_img).shape[0]):
                position = np.dot(H, np.array([x, y, 1]).reshape(3, 1))
                x_r = int(position[0, 0])
                y_r = int(position[1, 0])
                self.gray_and_a_chanel_target_img[y, x, 1] = self.gray_and_a_chanel_template_img[y_r, x_r, 1]

        # morphological operation, we need closing first dilation then erosion
        self.gray_and_a_chanel_target_img = self.morphological_operation(self.gray_and_a_chanel_target_img, 1)
        self.gray_and_a_chanel_target_img = self.morphological_operation(self.gray_and_a_chanel_target_img, 0)
        self.gray_and_a_chanel_template_img = self.morphological_operation(self.gray_and_a_chanel_template_img, 1)
        self.gray_and_a_chanel_template_img = self.morphological_operation(self.gray_and_a_chanel_template_img, 0)

        plt.figure("exact field")
        plt.subplot(121)
        plt.imshow(self.gray_and_a_chanel_target_img[:, :, 0] * self.gray_and_a_chanel_target_img[:, :, 1],
                   cmap="gray")
        plt.title('left')

        plt.subplot(122)
        plt.imshow(self.gray_and_a_chanel_template_img[:, :, 0] * self.gray_and_a_chanel_template_img[:, :, 1],
                   cmap="gray")
        plt.title('right')

        plt.show()

    def morphological_operation(self, input_img, rad):
        # rad = 0 means erosion and rad = 1 means dilation
        new_img = np.array(input_img)
        # n counts x, and k counts y
        for n in range(1, np.array(input_img).shape[1] - 1, 1):
            for k in range(1, np.array(input_img).shape[0] - 1, 1):
                if rad == 0:
                    new_img[k, n, 1] = min(input_img[k, n, 1], input_img[k + 1, n, 1],
                                           input_img[k - 1, n, 1], input_img[k, n + 1, 1],
                                           input_img[k, n - 1, 1], input_img[k + 1, n - 1, 1],
                                           input_img[k + 1, n + 1, 1], input_img[k - 1, n + 1, 1],
                                           input_img[k - 1, n - 1, 1])
                elif rad == 1:
                    new_img[k, n, 1] = max(input_img[k, n, 1], input_img[k + 1, n, 1],
                                           input_img[k - 1, n, 1], input_img[k, n + 1, 1],
                                           input_img[k, n - 1, 1], input_img[k + 1, n - 1, 1],
                                           input_img[k + 1, n + 1, 1], input_img[k - 1, n + 1, 1],
                                           input_img[k - 1, n - 1, 1])
                else:
                    print('Please give right number, 0 means erosion and 1 means delation')
        return new_img

    def get_lapseTime(self, aTime, bTime):
        aNum = 3600 * int(aTime[:2]) + 60 * int(aTime[2:4]) + int(aTime[-2:])
        bNum = 3600 * int(bTime[:2]) + 60 * int(bTime[2:4]) + int(bTime[-2:])
        gapH = (bNum - aNum) // 3600
        gapM = ((bNum - aNum) % 3600) // 60
        gapS = ((bNum - aNum) % 3600) % 60
        gapTime = "%02d:%02d:%02d" % (gapH, gapM, gapS)
        return (gapTime)
