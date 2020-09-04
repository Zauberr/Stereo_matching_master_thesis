import numpy as np
import cv2

"""
Firstly, I don't know, how I can design a image to test which coordinate function
pyrDown uses.
Then, in my opinion, function pyrDown has first use gaussian kernel then easily remove every even-numbered row and column
so it does't matter, which coordinate it uses.
It uses the image matrix to calculate the result. And we don't need to see the exact image of each 
layer. we just need it to calculate the next parameter p. And for function warpPerspective we have to see the image now.
"""

img_2_transform = np.array([[100, 100, 170, 170], [100, 100, 170, 170],
                            [170, 170, 170, 170], [170, 170, 170, 170]], dtype=float)
cv2.imwrite('img_2_transform.png', img_2_transform)
img_form = cv2.pyrDown(img_2_transform)
cv2.imwrite('img_form.png', img_form)
