import numpy as np
import cv2

img_2_transform = np.array([[0, 1], [1, 1]])
cv2.imwrite('img_2_transform.png', img_2_transform)
H = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 1]])
img_from = cv2.warpPerspective(img_2_transform, H, (20, 20), flags=cv2.INTER_LINEAR)
cv2.imwrite('img_form.png', img_from)
