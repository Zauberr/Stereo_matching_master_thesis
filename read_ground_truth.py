import numpy as np
import os


def read_ground_truth(filename):
    f = open(filename, 'r')
    line = f.readlines()
    print(line)
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


if __name__ == "__main__":
    filename = 'ground_truth_number.txt'
    H_inf, e, q = read_ground_truth(filename)
    print(H_inf, e, q)
