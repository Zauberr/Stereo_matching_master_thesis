from scipy.spatial.transform import Rotation as R
from numpy import *
from numpy.linalg import *
from cv2 import *

# H = K2 * R2^T * [Id - (C1-C2) * (-n^T) / (n^T * (t-C1))] * R1 * K1^-1

# file:
# folderName = 'Built_stereo_image' + '/Normal_case_1'
# if not os.path.exists(folderName):
#     os.makedirs(folderName)

text_name = 'ground_truth.txt'
text = open(text_name, 'w+')
text_name2 = 'ground_truth_number.txt'
text_num = open(text_name2, 'w+')
# planes
# blender use extrinsic rotation parameter to rotate the object.
n = R.from_euler('xyz', [[90.0, 0.0464, 90.0],
                         [90.0, 90.0, 0.0],
                         [0.0, 0.0, -180.0]], degrees=True).apply([0.0, 0.0, 1.0])

t = array([[0.81313, -1.0, 0.0],
           [1.0, -0.73745, 0.0],
           [1.0, -1.0, -0.23014]])

# cameras
C = array([[8.0, -7.0, 5.0],
           [8.3214, -6.617, 5.0]])

# 1 is left camera, H transform left to right
r = R.from_euler('xyz', [[65.0, 0.0, 50.0],
                         [61.0, 0.0, 55.0]], degrees=True)

# camera internals f = focal_length * k = focal_length * 100
f = 1500
p = array([600, 400])

# calculation
K = diag([f, f, 1.0])
K[0:2, 2] = p
K = K @ diag([1.0, -1.0, -1.0])  # convert between right-up-backwards and right-down-forward CS

# H0 = R2^T * R1 - R2^T * (C1-C2) * (-n^T) / (n^T * (t-C1)) * R1 = R0 - e0 * (-n0^T)
R0 = r[1].inv() * r[0]
e0 = r[1].inv().apply(C[0] - C[1])
# I don't know why R1 is r[0].inv(), not r[0]
n0 = array([r[0].inv().apply(ni) / (ni @ (ti - C[0])) for ni, ti in zip(n, t)])

# Hoo = K * R0 * K^-1, e = K * e0, p^T = n0^T * K^-1
Hoo = K @ R0.as_matrix() @ inv(K)  # array 3 * 3
e = K @ e0  # array 3 *
p = n0 @ inv(K)  # array 3 * 3
print('Hoo:', Hoo)
print('e:', e)
print('p:', p)

print('Hoo:', file=text)
print(Hoo, file=text)
print('e:', file=text)
print(e, file=text)
print('p:', file=text)
print(p, file=text)

for i in range(0, 3):
    for j in range(0, 3):
        print(Hoo[i, j], file=text_num)

for i in range(0, 3):
    print(e[i], file=text_num)

for i in range(0, 3):
    for j in range(0, 3):
        print(p[i, j], file=text_num)

# H = Hoo + e * p^T
H = [Hoo + outer(e, pi) for pi in p]
print(H)
# image warping

target_img_R = imread('R.png', cv2.IMREAD_GRAYSCALE)
template_img_L = imread('L.png', cv2.IMREAD_GRAYSCALE)
img1 = warpPerspective(target_img_R, H[0], (1200, 800), flags=WARP_INVERSE_MAP)
img2 = warpPerspective(target_img_R, H[1], (1200, 800), flags=WARP_INVERSE_MAP)
img3 = warpPerspective(target_img_R, H[2], (1200, 800), flags=WARP_INVERSE_MAP)

target_name = 'Right.png'
template_name = 'Left.png'
result_name1 = 'Left1.png'
result_name2 = 'Left2.png'
result_name3 = 'Left3.png'

imwrite(target_name, target_img_R)
imwrite(template_name, template_img_L)
imwrite(result_name1, img1)
imwrite(result_name2, img2)
imwrite(result_name3, img3)

text.close()
text_num.close()
print('end')
