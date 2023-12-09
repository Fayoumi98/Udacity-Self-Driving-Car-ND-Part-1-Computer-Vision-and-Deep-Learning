import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


img_path = '/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 _ Computer Vision and Deep Learning/Module 04_ Computer Vision/Lesson 02: Gradients and Color Spaces/Sobel Operator/curved-lane.jpg'

im = mpimg.imread(img_path)

# You need to pass a single color channel to the cv2.Sobel() function, so first convert it to grayscale
# plt.imshow(gray,cmap='gray')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Calculate the derivative in the x direction (the 1, 0 at the end denotes x direction):
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
#Calculate the derivative in the y direction (the 0, 1 at the end denotes y direction):
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

# Calculate the absolute value of the x derivative:
abs_sobelx = np.absolute(sobelx)

'''
** Note: ** It's not entirely necessary to convert to 8-bit (range from 0 to 255) 
but in practice, it can be useful in the event that you've written a function to apply 
a particular threshold, and you want it to work the same on input images of different scales,
like jpg vs. png. You could just as well choose a different standard range of values, like 0 to 1 etc.
'''
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))



thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1


plt.imshow(sxbinary, cmap='gray')
plt.show()