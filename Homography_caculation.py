import numpy as np
import math
import cv2
import math


def rotation_matrix(w, x, y, z):
    # Rotation from world to camera
    Rotation = np.array([[(math.pow(w, 2) + math.pow(x, 2) - math.pow(y, 2) - math.pow(z, 2)), 2 * (x * y - w * z),
                          2 * (x * z + w * y)],
                         [2 * (x * y + w * z), (math.pow(w, 2) - math.pow(x, 2) + math.pow(y, 2) - math.pow(z, 2)),
                          2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x),
                          (math.pow(w, 2) - math.pow(x, 2) - math.pow(y, 2) + math.pow(z, 2))]])
    return Rotation


def transformation_matrix(rotation, positionOfCam):
    temp_0 = -np.matmul(np.transpose(rotation), positionOfCam)
    temp_1 = np.concatenate((np.transpose(rotation), temp_0), axis=1)
    transformation = np.concatenate((temp_1, np.array([[0, 0, 0, 1]])), axis=0)

    return transformation


def calibration_matrix(focal_length, scaling_factor, width, height):
    # use middle of the pixel in the upper left corner as the original of the Camera CS
    K = np.array([[(- focal_length * scaling_factor), 0, (width / 2) - 0.5, 0],
                  [0, (focal_length * scaling_factor), (height / 2) - 0.5, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    K_small = np.array([[(- focal_length * scaling_factor), 0, (width / 2) - 0.5],
                        [0, (focal_length * scaling_factor), (height / 2) - 0.5],
                        [0, 0, 1]])
    return K, K_small


def homography_matrix_R_2_L(H_L, H_R, axis):
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


def normal_vector_of_plane(w, x, y, z):
    rotation = rotation_matrix(w, x, y, z)
    normal_vector = np.matmul(rotation, np.array([[0],
                                                  [0],
                                                  [1]]))
    return normal_vector


def distance_from_plane_to_camera(normal_vector, point_on_plane, camera_center1):
    d_numerator = abs(normal_vector.T @ camera_center1 - normal_vector.T @ point_on_plane)
    d_denominator = np.linalg.norm(normal_vector)
    d = d_numerator / d_denominator
    return d


if __name__ == "__main__":
    text = open('/Users/zauber/Desktop/Built_stereo_image/Normal_case/result.txt', 'w+')

    # right to left. 1: right 2:left and the Rotation in H_infty calculation should use rotation_L^T
    # c is the same

    rotation_L = rotation_matrix(0.764, 0.487, 0.227, 0.356)
    rotation_L_T = rotation_L.T
    rotation_L_in = np.linalg.inv(rotation_L)
    rotation_R = rotation_matrix(0.764, 0.450, 0.234, 0.398)

    c_r = np.array([[8.3214],
                    [-6.617],
                    [5]])
    c_l = np.array([[8],
                    [-7],
                    [5]])

    transformation_L = transformation_matrix(rotation_L, c_l)
    transformation_R = transformation_matrix(rotation_R, c_r)

    # print("transformation_matrix_L:", transformation_L)
    K, K_small = calibration_matrix(0.03, 99740, 2304, 1536)
    # print("calibration_matrix:", K)

    # H is the homography from the right image to left image.
    H_L = np.matmul(K, transformation_L)
    H_R = np.matmul(K, transformation_R)

    # Homography for brick wall, X = 0
    Homography_of_brickwall = homography_matrix_R_2_L(np.linalg.inv(H_L), np.linalg.inv(H_R), "x")
    Homography_of_checkboard = homography_matrix_R_2_L(np.linalg.inv(H_L), np.linalg.inv(H_R), "y")
    Homography_of_markovfractaltexture = homography_matrix_R_2_L(np.linalg.inv(H_L), np.linalg.inv(H_R), "z")

    print("Homography_of_brickwall:")
    print(Homography_of_brickwall)
    print("Homography_of_checkboard:")
    print(Homography_of_checkboard)
    print("Homography_of_markovfractaltexture:")
    print(Homography_of_markovfractaltexture)

    print("Homography_of_brickwall:", file=text)
    print(Homography_of_brickwall, file=text)
    print("Homography_of_checkboard:", file=text)
    print(Homography_of_checkboard, file=text)
    print("Homography_of_markovfractaltexture:", file=text)
    print(Homography_of_markovfractaltexture, file=text)

    # testing
    template_img_R = cv2.imread('/Users/zauber/Desktop/Built_stereo_image/Normal_case/R.png', cv2.IMREAD_GRAYSCALE)
    target_img_L = cv2.imread('/Users/zauber/Desktop/Built_stereo_image/Normal_case/L.png', cv2.IMREAD_GRAYSCALE)

    template_img_R = np.float32(template_img_R)
    target_img_L = np.float32(target_img_L)

    width_x = np.array(template_img_R).shape[1]
    height_y = np.array(template_img_R).shape[0]


    # brickwall. I got warped image to compare with template image.
    warped_img_of_brickwall = cv2.warpPerspective(target_img_L, Homography_of_brickwall, (width_x, height_y),
                                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # checkboard
    warped_img_of_Voronoi = cv2.warpPerspective(target_img_L, Homography_of_checkboard, (width_x, height_y),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Markov fractal texture
    warped_img_of_Markov = cv2.warpPerspective(target_img_L, Homography_of_markovfractaltexture, (width_x, height_y),
                                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    '''
    brick wall:
    Rectangle Selection
      X: 936
      Y: 213
      Width: 333
      Height: 324
    
    Voronoi Texture:
    Rectangle Selection
      X: 1563
      Y: 309
      Width: 267
      Height: 372
    
    Markov Texture:
    Rectangle Selection
      X: 1020
      Y: 867
      Width: 366
      Height: 171
    [x, width, y, height]
    '''

    brickwall_field = [936, 333, 213, 324]
    Voronoi_field = [1563, 267, 309, 372]
    Markov_field = [1020, 366, 867, 171]

    cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Normal_case/template_img.png', template_img_R)

    cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Normal_case/template_brickwall.png',
                template_img_R[brickwall_field[2]:(brickwall_field[2] + brickwall_field[3]),
                brickwall_field[0]:brickwall_field[0] + brickwall_field[1]])
    cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Normal_case/template_Voronoi.png',
                template_img_R[Voronoi_field[2]:(Voronoi_field[2] + Voronoi_field[3]),
                Voronoi_field[0]:Voronoi_field[0] + Voronoi_field[1]])
    cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Normal_case/template_Markov.png',
                template_img_R[Markov_field[2]:(Markov_field[2] + Markov_field[3]),
                Markov_field[0]:Markov_field[0] + Markov_field[1]])

    cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Normal_case/warped_img_of_brickwall.png',
                warped_img_of_brickwall[brickwall_field[2]:(brickwall_field[2] + brickwall_field[3]),
                brickwall_field[0]:brickwall_field[0] + brickwall_field[1]])
    cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Normal_case/warped_img_of_Voronoi.png',
                warped_img_of_Voronoi[Voronoi_field[2]:(Voronoi_field[2] + Voronoi_field[3]),
                Voronoi_field[0]:Voronoi_field[0] + Voronoi_field[1]])
    cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Normal_case/warped_img_of_Markov.png',
                warped_img_of_Markov[Markov_field[2]:(Markov_field[2] + Markov_field[3]),
                Markov_field[0]:Markov_field[0] + Markov_field[1]])

    # H_infty and e calculation
    H_infty = K_small @ rotation_L.T @ np.linalg.inv(rotation_R.T) @ np.linalg.inv(K_small)
    e = K_small @ rotation_L.T @ (c_r - c_l)

    # q_n calculation
    # way 1: the positive way

    # normal vector of plane

    n_brickwall = normal_vector_of_plane(0.5, 0.5, 0.5, 0.5)
    n_checkboard = normal_vector_of_plane(0.5, 0.5, 0.5, -0.5)
    n_markov = normal_vector_of_plane(0.0, 0.0, 0.0, 1.0)

    # distance from plane to camera center 1(right)
    d_brickwall = distance_from_plane_to_camera(n_brickwall, np.array([[0],
                                                                       [0],
                                                                       [0]], dtype=float), c_r)
    d_checkboard = distance_from_plane_to_camera(n_checkboard, np.array([[1],
                                                                         [0],
                                                                         [0]], dtype=float), c_r)
    d_markov = distance_from_plane_to_camera(n_markov, np.array([[1],
                                                                 [-1],
                                                                 [0]], dtype=float), c_r)

    q_brickwall_T = n_brickwall.T @ np.linalg.inv(rotation_R.T) @ np.linalg.inv(K_small) / d_brickwall
    q_checkboard_T = n_checkboard.T @ rotation_R @ np.linalg.inv(K_small) / d_checkboard
    q_markov_T = n_markov.T @ rotation_R @ np.linalg.inv(K_small) / d_markov

    recalculate_H_brick = H_infty + e @ q_brickwall_T
    recalculate_H_check = H_infty + e @ q_checkboard_T
    recalculate_H_markov = H_infty + e @ q_markov_T

    # print in file
    print('q_brickwall:', q_brickwall_T.T)
    print('q_checkboard:', q_checkboard_T.T)
    print('q_markov:', q_markov_T.T)
    print('H_infty:', H_infty)
    print('e:', e)

    # instant output
    print('H_infty:', H_infty, file=text)
    print('e:', e, file=text)
    print('q_brickwall:', q_brickwall_T.T, file=text)
    print('q_checkboard:', q_checkboard_T.T, file=text)
    print('q_markov:', q_markov_T.T, file=text)

    print('recalculate_H_brick:', recalculate_H_brick)
    print('recalculate_H_check:', recalculate_H_check)
    print('recalculate_H_markov:', recalculate_H_markov)

    print('recalculate_H_brick:', recalculate_H_brick, file=text)
    print('recalculate_H_check:', recalculate_H_check, file=text)
    print('recalculate_H_markov:', recalculate_H_markov, file=text)

    difference_H_1 = Homography_of_brickwall @ np.linalg.inv(recalculate_H_brick)
    difference_H_2 = Homography_of_checkboard @ np.linalg.inv(recalculate_H_check)
    difference_H_3 = Homography_of_markovfractaltexture @ np.linalg.inv(recalculate_H_markov)

    print(difference_H_1)
    print(difference_H_2)
    print(difference_H_3)
    # # brickwall. I got warped image to compare with template image.
    # warped_img_of_brickwall = cv2.warpPerspective(target_img_L, recalculate_H_brick, (width_x, height_y),
    #                                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    #
    # # checkboard
    # warped_img_of_Voronoi = cv2.warpPerspective(target_img_L, recalculate_H_check, (width_x, height_y),
    #                                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    #
    # # Markov fractal texture
    # warped_img_of_Markov = cv2.warpPerspective(target_img_L, recalculate_H_markov, (width_x, height_y),
    #                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Normal_case/template_img.png', template_img_R)
    #
    # cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Normal_case/template_brickwall.png',
    #             template_img_R[brickwall_field[2]:(brickwall_field[2] + brickwall_field[3]),
    #             brickwall_field[0]:brickwall_field[0] + brickwall_field[1]])
    # cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Normal_case/template_Voronoi.png',
    #             template_img_R[Voronoi_field[2]:(Voronoi_field[2] + Voronoi_field[3]),
    #             Voronoi_field[0]:Voronoi_field[0] + Voronoi_field[1]])
    # cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Normal_case/template_Markov.png',
    #             template_img_R[Markov_field[2]:(Markov_field[2] + Markov_field[3]),
    #             Markov_field[0]:Markov_field[0] + Markov_field[1]])
    #
    # cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Normal_case/warped_img_of_brickwall.png',
    #             warped_img_of_brickwall[brickwall_field[2]:(brickwall_field[2] + brickwall_field[3]),
    #             brickwall_field[0]:brickwall_field[0] + brickwall_field[1]])
    # cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Normal_case/warped_img_of_Voronoi.png',
    #             warped_img_of_Voronoi[Voronoi_field[2]:(Voronoi_field[2] + Voronoi_field[3]),
    #             Voronoi_field[0]:Voronoi_field[0] + Voronoi_field[1]])
    # cv2.imwrite('/Users/zauber/Desktop/Built_stereo_image/Normal_case/warped_img_of_Markov.png',
    #             warped_img_of_Markov[Markov_field[2]:(Markov_field[2] + Markov_field[3]),
    #             Markov_field[0]:Markov_field[0] + Markov_field[1]])

    text.close()
    print('end')
