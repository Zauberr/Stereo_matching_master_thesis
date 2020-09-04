import calibrated_img as cali
import time
import cv2
import numpy as np

recordTime = time.time()
startTime = time.strftime("%H%M%S")

cross_image = cv2.imread('Photo/cross.pgm')
height, width = np.array(cross_image).shape[0:2]
H = np.array([[1.1, 0.05, -10], [0, 1, 0], [0, 0, 1]], dtype=float)
print(H)
transformed_cross_img = cv2.warpPerspective(cross_image, H, (width, height))
cv2.imwrite('Photo/transformed_cross_img.png', transformed_cross_img)

test = cali.calibrated_image('Photo/cross.pgm', 'Photo/transformed_cross_img.png')
a, b, c = (0.1, 0.05, -8)
img_warped = test.warp_image_L2R((a, b, c))
Delta_p = test.improve_parameter(img_warped, [26, 44, 31, 41])

# Rectangle Selection
#   X: 26
#   Y: 32
#   Width: 42
#   Height: 37


cv2.imwrite('first_warped_img.png', img_warped)

# cv2.namedWindow('test')
# cv2.imshow('test', img_warped)
# cv2.waitKey(2)
# print(Delta_p)
count = 0
while abs(Delta_p[0, 0]) > 0.01 or abs(Delta_p[1, 0]) > 0.01 or abs(Delta_p[2, 0]) > 0.01:
    count = count + 1
    if count > 50:
        print('more than 50 cycles')
        break

    a = a + Delta_p[0, 0]
    b = b + Delta_p[1, 0]
    c = c + Delta_p[2, 0]

    img_warped = test.warp_image_L2R((a, b, c))
    Delta_p = test.improve_parameter(img_warped, [26, 44, 31, 41])

print('cycles:', count)
print('Delta_p = ', Delta_p)
print('Parameter: ', a, b, c)
# test.show_difference_of_warp_image(image_warp, [500, 900, 20, 200])

cv2.imwrite('from_h_img.png', img_warped)

timeGap = time.time() - recordTime
if timeGap >= 1:  # 这是按1秒设置的，可以根据实际需要设置
    recordTime += timeGap
    showTime_String = test.get_lapseTime(startTime, time.strftime("%H%M%S"))
    print('run time :', showTime_String)
