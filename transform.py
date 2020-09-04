from scipy.spatial.transform import Rotation as R
from numpy import *
from numpy.linalg import *
from cv2 import *

# H = K2 * R2^T * [Id - (C1-C2) * (-n^T) / (n^T * (t-C1))] * R1 * K1^-1

# planes
n = R.from_euler('XYZ', [[30.0, 30.0, 0.0],
                         [0.0, -45.0, -20.0],
                         [-60.0, 0.0, -10.0]], degrees=True).apply([0.0, 0.0, 1.0])

t = array([[-200.0, 150.0, 0.0],
           [200.0, 50.0, 0.0],
           [-125.0, -150.0, 200.0]])

# cameras
C = array([[-500.0, 500.0, 1500.0],
           [500.0, 500.0, 1500.0]])

r = R.from_euler('XYZ', [[-20.0, -15.0, -10.0],
                         [-20.0, 20.0, 10.0]], degrees=True)

# camera internals
f = 1000.0
p = array([399.5, 299.5])

# calculation
K = diag([f, f, 1.0])
K[0:2, 2] = p
K = K @ diag([1.0, -1.0, -1.0])  # convert between right-up-backwards and right-down-forward CS

# H0 = R2^T * R1 - R2^T * (C1-C2) * (-n^T) / (n^T * (t-C1)) * R1 = R0 - e0 * (-n0^T)
R0 = r[1].inv() * r[0]
e0 = r[1].inv().apply(C[0] - C[1])
n0 = array([r[0].inv().apply(ni) / (ni @ (ti - C[0])) for ni, ti in zip(n, t)])

# Hoo = K * R0 * K^-1, e = K * e0, p^T = -n0^T * K^-1
Hoo = K @ R0.as_matrix() @ inv(K)
e = K @ e0
p = -n0 @ inv(K)

# H = Hoo - e * p^T
H = [Hoo - outer(e, pi) for pi in p]

# image warping
img = imread("Photo/right.png", IMREAD_UNCHANGED)
img1 = warpPerspective(img, H[0], (800, 600), flags=WARP_INVERSE_MAP)
img2 = warpPerspective(img, H[1], (800, 600), flags=WARP_INVERSE_MAP)
img3 = warpPerspective(img, H[2], (800, 600), flags=WARP_INVERSE_MAP)
imwrite("left1.png", img1)
imwrite("left2.png", img2)
imwrite("left3.png", img3)
