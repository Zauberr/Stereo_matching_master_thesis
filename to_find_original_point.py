import numpy as np
import cv2

img_2_transform = np.array([[100, 170], [170, 170]])
cv2.imwrite('img_2_transform.png', img_2_transform)
H = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
img_warped = cv2.warpPerspective(img_2_transform, H, (4, 4), flags=cv2.INTER_NEAREST)

cv2.imwrite('img_warped.png', img_warped)
