import numpy as np
import math
import cv2


def rotation_matrix(w, x, y, z):
    Rotation = np.array([[(math.pow(w, 2) + math.pow(x, 2) - math.pow(y, 2) - math.pow(z, 2)), 2 * (x * y - w * z),
                          2 * (x * z + w * y)],
                         [2 * (x * y + w * z), (math.pow(w, 2) - math.pow(x, 2) + math.pow(y, 2) - math.pow(z, 2)),
                          2 * (y * z - w * x)],
                         [2 * (x * y - w * z), 2 * (y * z + w * x),
                          (math.pow(w, 2) - math.pow(x, 2) - math.pow(y, 2) + math.pow(z, 2))]])
    # R_z = np.array([[math.cos(math.radians(angle_z)), -math.sin(math.radians(angle_z)), 0],
    #                 [math.sin(math.radians(angle_z)), math.cos(math.radians(angle_z)), 0],
    #                 [0, 0, 1]])
    # R_y = np.array([[math.cos(math.radians(angle_y)), 0, math.sin(math.radians(angle_y))],
    #                 [0, 1, 0],
    #                 [-math.sin(math.radians(angle_y)), 0, math.cos(math.radians(angle_y))]])
    # R_x = np.array([[1, 0, 0],
    #                 [0, math.cos(math.radians(angle_x)), -math.sin(math.radians(angle_x))],
    #                 [0, math.sin(math.radians(angle_x)), math.cos(math.radians(angle_x))]])
    # R = R_z @ R_y @ R_x

    return Rotation


def transformation_matrix(rotation, positionOfCam):
    temp_0 = -np.matmul(np.transpose(rotation), positionOfCam)
    temp_1 = np.concatenate((np.transpose(rotation), temp_0), axis=1)
    transformation = np.concatenate((temp_1, np.array([[0, 0, 0, 1]])), axis=0)

    return transformation


def calibration_matrix(focal_length, scaling_factor, width, height):
    K = np.array([[(- focal_length * scaling_factor), 0, (width / 2) - 0.5, 0],
                  [0, (focal_length * scaling_factor), (height / 2) - 0.5, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return K


def homography_matrix(H_L, H_R, axis):
    if axis == "x":
        # left image
        H_L1 = H_L[1:4, 0:3]
        h_L4 = H_L[1:4, 3].reshape(3, 1)
        h_L1 = H_L[0, 0:3] / H_L[0, 3]
        h_L1 = h_L1.reshape(3, 1)
        H_L_rebuilt = np.matmul(np.concatenate((H_L1, h_L4), axis=1),
                                np.concatenate((np.eye(3), -np.transpose(h_L1)), axis=0))
        # right image
        H_R1 = H_R[1:4, 0:3]
        h_R4 = H_R[1:4, 3].reshape(3, 1)
        h_R1 = H_R[0, 0:3] / H_R[0, 3]
        h_R1 = h_R1.reshape(3, 1)
        H_R_rebuilt = np.matmul(np.concatenate((H_R1, h_R4), axis=1),
                                np.concatenate((np.eye(3), -np.transpose(h_R1)), axis=0))
        H_R_to_L = np.matmul(np.linalg.inv(H_L_rebuilt), H_R_rebuilt)
    elif axis == "y":
        # left image
        H_L1 = np.concatenate((H_L[0, 0:3].reshape(1, 3), H_L[2:4, 0:3]), axis=0)
        h_L4 = np.array([[H_L[0, 3]],
                         [H_L[2, 3]],
                         [H_L[3, 3]]])
        h_L2 = H_L[1, 0:3] / H_L[1, 3]
        h_L2 = h_L2.reshape(3, 1)
        H_L_rebuilt = np.matmul(np.concatenate((H_L1, h_L4), axis=1),
                                np.concatenate((np.eye(3), -np.transpose(h_L2)), axis=0))
        # right image
        H_R1 = np.concatenate((H_R[0, 0:3].reshape(1, 3), H_R[2:4, 0:3]), axis=0)
        h_R4 = np.array([[H_R[0, 3]],
                         [H_R[2, 3]],
                         [H_R[3, 3]]])
        h_R2 = H_R[1, 0:3] / H_R[1, 3]
        h_R2 = h_R2.reshape(3, 1)
        H_R_rebuilt = np.matmul(np.concatenate((H_R1, h_R4), axis=1),
                                np.concatenate((np.eye(3), -np.transpose(h_R2)), axis=0))
        H_R_to_L = np.matmul(np.linalg.inv(H_L_rebuilt), H_R_rebuilt)
    elif axis == "z":
        # left image
        H_L1 = np.concatenate((H_L[0:2, 0:3], H_L[3, 0:3].reshape(1, 3)), axis=0)
        h_L4 = np.array([[H_L[0, 3]],
                         [H_L[1, 3]],
                         [H_L[3, 3]]])
        h_L3 = H_L[2, 0:3] / H_L[2, 3]
        h_L3 = h_L3.reshape(3, 1)
        H_L_rebuilt = np.matmul(np.concatenate((H_L1, h_L4), axis=1),
                                np.concatenate((np.eye(3), -np.transpose(h_L3)), axis=0))
        # right image
        H_R1 = np.concatenate((H_R[0:2, 0:3], H_R[3, 0:3].reshape(1, 3)), axis=0)
        h_R4 = np.array([[H_R[0, 3]],
                         [H_R[1, 3]],
                         [H_R[3, 3]]])
        h_R3 = H_R[2, 0:3] / H_R[2, 3]
        h_R3 = h_R3.reshape(3, 1)
        H_R_rebuilt = np.matmul(np.concatenate((H_R1, h_R4), axis=1),
                                np.concatenate((np.eye(3), -np.transpose(h_R3)), axis=0))
        H_R_to_L = np.matmul(np.linalg.inv(H_L_rebuilt), H_R_rebuilt)
    else:
        print("The axis is wrong")

    return H_R_to_L


if __name__ == "__main__":
    text = open('/Users/zauber/Desktop/Built_stereo_image/Simple_case/text.txt', 'w+')

    rotation_L = rotation_matrix(90, 0, 90)
    rotation_R = rotation_matrix(90, 0, 90)

    transformation_L = transformation_matrix(rotation_L, np.array([[5],
                                                                   [-0.5],
                                                                   [0]]))
    transformation_R = transformation_matrix(rotation_R, np.array([[5],
                                                                   [0.5],
                                                                   [0]]))

    # print("transformation_matrix_L:", transformation_L)
    K = calibration_matrix(0.03, 100000, 2304, 1536)
    # print("calibration_matrix:", K)

    # H is the homography from the right image to left image.
    H_L = np.matmul(K, transformation_L)
    H_R = np.matmul(K, transformation_R)

    # Homography for brick wall, X = 0
    Homography_of_brickwall = homography_matrix(np.linalg.inv(H_L), np.linalg.inv(H_R), "x")

    print("Homography_of_brickwall:")
    print(Homography_of_brickwall)

    print("Homography_of_brickwall:", file=text)
    print(Homography_of_brickwall, file=text)

    # testing
    template_img_R = cv2.imread('/Users/zauber/Desktop/Built_stereo_image/Simple_case/simple_case_R.png',
                                cv2.IMREAD_GRAYSCALE)
    target_img_L = cv2.imread('/Users/zauber/Desktop/Built_stereo_image/Simple_case/simple_case_L.png',
                              cv2.IMREAD_GRAYSCALE)

    template_img_R = np.float32(template_img_R)
    target_img_L = np.float32(target_img_L)

    width_x = np.array(template_img_R).shape[1]
    height_y = np.array(template_img_R).shape[0]

    # brickwall. I got warped image to compare with template image(right image).
    warped_img_of_brickwall = cv2.warpPerspective(target_img_L, Homography_of_brickwall, (width_x, height_y),
                                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Simple_case/warped_img_of_brickwall.png',
                warped_img_of_brickwall)
    cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Simple_case/template_img.png', template_img_R)

    text.close()
    print('end')
'''
if __name__ == "__main__":
    rotation_L = rotation_matrix(0, 0, 0)
    rotation_R = rotation_matrix(0, 0, 0)

    transformation_L = transformation_matrix(rotation_L, np.array([[-0.5],
                                                                   [0],
                                                                   [5]]))
    transformation_R = transformation_matrix(rotation_R, np.array([[0.5],
                                                                   [0],
                                                                   [5]]))

    # print("transformation_matrix_L:", transformation_L)
    K = calibration_matrix(0.03, 100000, 2304, 1536)
    # print("calibration_matrix:", K)

    # H is the homography from the right image to left image.
    H_L = np.matmul(K, transformation_L)
    H_R = np.matmul(K, transformation_R)

    # Homography for brick wall, X = 0
    Homography_of_brickwall = homography_matrix(np.linalg.inv(H_L), np.linalg.inv(H_R), "z")

    print("Homography_of_brickwall:")
    print(Homography_of_brickwall)

    # testing
    template_img_R = cv2.imread('/Users/zauber/Desktop/Built_stereo_image/with_new_resolution_R.png', cv2.IMREAD_GRAYSCALE)
    target_img_L = cv2.imread('/Users/zauber/Desktop/Built_stereo_image/with_new_resolution_L.png', cv2.IMREAD_GRAYSCALE)

    template_img_R = np.float32(template_img_R)
    target_img_L = np.float32(target_img_L)

    width_x = np.array(template_img_R).shape[1]
    height_y = np.array(template_img_R).shape[0]

    # brickwall. I got warped image to compare with template image(right image).
    warped_img_of_brickwall = cv2.warpPerspective(target_img_L, Homography_of_brickwall, (width_x, height_y),
                                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    cv2.imwrite('warped_img_of_brickwall.png', warped_img_of_brickwall)
    cv2.imwrite('template_img.png', template_img_R)
    print('end')
'''
