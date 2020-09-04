import numpy as np
import math


def calculate_RMSD(H_inf_gt, e_gt, q_gt, H_inf, e, q, fieldNum, field):
    sum = 0
    for i in range(fieldNum):  # i patch
        field_x = field[i]
        H_gt = H_inf_gt + e @ q_gt[i]
        H = H_inf + e @ q[i]
        D = 0
        n = 0
        d_max = 0
        for x in range(field_x[0], field_x[0] + field_x[1]):  # j pixel column
            for y in range(field_x[2], field_x[2] + field_x[3]):  # i pixel row
                x_ij = np.array([[x],
                                 [y],
                                 [1]])

                y_ij_gt_3D = H_gt @ x_ij
                y_ij_gt_2D = np.array([[y_ij_gt_3D[0] / y_ij_gt_3D[2]],
                                       [y_ij_gt_3D[1] / y_ij_gt_3D[2]]])

                y_ij_3D = H_gt @ x_ij
                y_ij_2D = np.array([[y_ij_3D[0] / y_ij_3D[2]],
                                    [y_ij_3D[1] / y_ij_3D[2]]])

                d = np.linalg.norm(y_ij_gt_2D - y_ij_2D)
                if d > d_max:
                    d_max = d
                D = D + d ** 2
                n = n + 1

    RMSD = math.sqrt(D / n)
    return RMSD, d_max
