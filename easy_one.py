import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# read image
img_left = cv2.imread('Photo/wide-left-rectified-8.jpg')
img_right = cv2.imread('Photo/wide-right-rectified-8.jpg')  # numpy array

# change to gray value
[b_l, g_l, r_l] = [img_left[:, :, i] for i in range(3)]
[b_r, g_r, r_r] = [img_right[:, :, i] for i in range(3)]
gray_l = 0.114 * b_l + 0.587 * g_l + 0.299 * r_l
gray_r = 0.114 * b_r + 0.587 * g_r + 0.299 * r_r

print(b_l, b_l.shape)
print(gray_l, gray_l.shape)

plt.subplot(121)
plt.imshow(gray_l, cmap="gray")

plt.subplot(122)
plt.imshow(gray_r, cmap="gray")

plt.show()

gray_r = gray_r.astype(np.uint8)  # float in unit8 changing

cv2.namedWindow('right', flags=0)
cv2.imshow('right', gray_r)

gray_l = gray_l.astype(np.uint8)
cv2.namedWindow('left', flags=0)
cv2.imshow('left', gray_l)

cv2.waitKey()
'''
#parameter definition
corr = [0 for n in range(20)]     #first I use numpy array but I don't know the data structure and I can't handle it
x_l_corr = [[0] for n in range(146)]
y_l_corr = [[0] for n in range(244)]
x_r = [[n] for n in range(475,621,1)]
print('x_r',x_r)
y_r=2

for x_r in range(475, 621, 1):              #for the plane area in every column
        n=0                                          #cout
        for x_l in range((x_r-10), (x_r+10),1):        #x_l in area [x_r-10, x_r+10] to find

            corr[n] = np.corrcoef(gray_r[y_r, (x_r-5):(x_r+5)] , gray_l[y_r, (x_l-5):(x_l+5)])[0, 1]
            n=n+1
        ind = corr.index(max(corr))
        x_l_corr[x_r-475][0]= ind+x_r-10
print('x_l_corr',x_l_corr)

y=np.array(x_r)
x=np.array(x_l_corr)
'''
'''
model = LinearRegression(fit_intercept=True,copy_X=True,n_jobs=1,normalize=False)
model.fit(x,y)
print(model.intercept_)
'''

# gray_l_corresponding[x_r,y_r]=zip(y_r, (x_r-10+np.argmax(corr)) #矩阵的一个点内容是一个坐标
