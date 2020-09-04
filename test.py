import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import calibrated_img as cali
import os

target_img = cv2.imread('Photo/wide-left-rectified-8.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('original image.png', target_img)
# blur the image
target_img = cv2.GaussianBlur(target_img, (7, 7), 20)
cv2.imwrite('blurred image.png', target_img)

a = np.array([[-1, 2, 3],
              [-4, 2, 0]])
b = abs(a)
c = np.max(b)
d = np.max(a)
print('c:', c)
print('d:', d)
'''
# M_1 is the part for Delta p in A_matrix. Dimension is 1 * 3
M_1 = np.array([[1, 2, 3]])

# M_2 is the part for Delta a and Delta b in A_matrix. Dimension is 1 * 2
M_2 = np.array([[4, 5]])

# M is the factor of variable
M = np.concatenate((M_1, M_2), axis=1)  # make sure
print(-M.T)
print(M[0, 3:5].reshape(2))
'''
# img_left = cv2.imread('Photo/wide-left-rectified-8.jpg')
# img_right = cv2.imread('Photo/wide-right-rectified-8.jpg')
#
# [b_l, g_l, r_l] = [img_left[:, :, i] for i in range(3)]
# [b_r, g_r, r_r] = [img_right[:, :, i] for i in range(3)]
# gray_l = 0.114 * b_l + 0.587 * g_l + 0.299 * r_l
# gray_r = 0.114 * b_r + 0.587 * g_r + 0.299 * r_r
# print('gray_l shape', np.array(gray_l).shape, 'gray_r shape',
#       np.array(gray_r).shape, end='\n')
#
# corr = np.zeros((100,))  # first i use numpy array but i don't know the data structure and i can't handle it
# x_r_used = []
# x_and_y_l = []
# # ind_img = np.zeros((245, 237))
# # in matrix y means rows, x means column. don't mix it
#
# # definition the corresponding region
# x_min = 428
# x_max = 655
# y_min = 0
# y_max = 246
#
# for y_l in range(y_min, y_max, 1):
#     for x_r in range(x_min, x_max, 1):
#         # for the plane area in every column
#         n = 0  # cout
#         for x_l in range(x_r, (x_r + 100), 1):  # x_l in area [x_r-10, x_r+10] to find
#
#             corr[n] = np.corrcoef(gray_r[y_l, (x_r - 10):(x_r + 11)],
#                                   gray_l[y_l, (x_l - 10):(x_l + 11)])[0, 1]
#             n = n + 1
#         # print(corr)
#         ind = (np.argmax(corr))
#         # print(ind)
#         # if (x_r == 420):
#         # break
#         # x_l_corr[x_r-475]= ind+x_r-10
#         x_and_y_l.append([y_l, (ind + x_r)])
#         x_r_used.append(x_r)
#         # ind_img[y_l-1, x_r-418] = ind/100
# # print('x_l_corr',x_l_corr)
#
#
# x_r_array = np.array(x_r_used)
# # x=np.array(x_l_corr)
# x_and_y_array = np.array(x_and_y_l)
#
# # x_res = x_and_y_array.reshape((146,2))
#
#
# model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=1, normalize=False)
# model.fit(x_and_y_array, x_r_array)
# c = model.intercept_
# b, a = model.coef_
# a = a - 1
# print('a =', a, 'b =', b, 'c =', c, end='\n')

# extract corresponding patch
# gray_l_patch_1 = gray_l[0:246, 418:655]
# gray_r_patch_1 = gray_r[0:246, 418:655]
# print('shape of gray_l_patch_1 and gray_r_patch_1 is ', np.array(gray_l_patch_1).shape)

# wrap image_l to right
# p = np.array([round(a, 3), round(b, 3), round(c, 3)]).reshape(1, 3)
# e1 = np.array([1, 0, 0]).reshape(3, 1)
# I = np.eye(3)
# H = I + np.dot(e1, p)
# print('p=', p)
# print('e1=', e1)
# print('H=', H)
#
# image_wrap = cv2.warpPerspective(gray_l, H, (768, 1024))
# print('image_wrap shape', np.array(image_wrap).shape)
#
# plt.subplot(221)
# plt.imshow(gray_l[y_min:y_max, x_min:x_max], cmap="gray")
# plt.title('left image')
#
# plt.subplot(222)
# plt.imshow(gray_r[y_min:y_max, x_min:x_max], cmap="gray")
# plt.title('right image')
# plt.subplot(224)
# plt.imshow(image_wrap[y_min:y_max, x_min:x_max], cmap="gray")
# plt.title('transform image')
# # plt.subplot(224)
# # plt.title('ind')
# # plt.imshow(ind_img)
#
#
# plt.show()
#
# # Improve p
# x_min_improve = 450
# x_max_improve = 650
# y_min_improve = 0
# y_max_improve = 220
#
# d_x_image_out = np.zeros(((y_max_improve - y_min_improve) * (x_max_improve - x_min_improve), 1))
# A_matrix = np.zeros(((y_max_improve - y_min_improve) * (x_max_improve - x_min_improve), 3))
# b_matrix = np.zeros(((y_max_improve - y_min_improve) * (x_max_improve - x_min_improve), 1))
# print('shape of A_matrix:', np.shape(A_matrix))
# count = 0
# for y in range(y_min_improve, y_max_improve, 1):
#     for x in range(x_min_improve, x_max_improve, 1):
#         d_x_image_out[count] = (image_wrap[y, x - 1] - image_wrap[y, x + 1]) / 2
#         A_matrix[count, :] = [x, y, 1] * d_x_image_out[count]
#         # A_matrix[count, 1] = y * d_x_image_out[count]
#         # A_matrix[count, 2] = d_x_image_out[count]
#         b_matrix[count] = gray_l[y, x] - image_wrap[y, x]
#         count = count + 1
# Delta_p = np.dot(np.dot(np.dot(np.transpose(A_matrix), A_matrix),
#                         np.transpose(A_matrix)), b_matrix)
# print(Delta_p)

'''
gray_r = gray_r.astype(np.uint8)
cv2.namedwindow('win', flags=0)
cv2.imshow('win', gray_r)
cv2.waitkey()
'''
