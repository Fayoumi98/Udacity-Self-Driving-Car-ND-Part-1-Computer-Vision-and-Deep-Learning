import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


img_path = '/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 _ Computer Vision and Deep Learning/Module 04_ Computer Vision/Lesson 02: Gradients and Color Spaces/Magnitude of the Gradient/curved-lane.jpg'
im = mpimg.imread(img_path)

# You need to pass a single color channel to the cv2.Sobel() function, so first convert it to grayscale
# plt.imshow(gray,cmap='gray')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Calculate the derivative in the x direction (the 1, 0 at the end denotes x direction):
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
#Calculate the derivative in the y direction (the 0, 1 at the end denotes y direction):
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

# Calculate the abbsolute value of the x derivative:
abs_sobelx = np.absolute(sobelx)
abs_sobely = np.absolute(sobely)

'''
The magnitude, or absolute value, of the gradient is just the square root of 
the squares of the individual x and y gradients. For a gradient in both the 
x and y directions, the magnitude is the square root of the sum of the squares.
'''
gradmag = np.sqrt(sobelx**2 + sobely**2)


# Rescale to 8 bit
scale_factor = np.max(gradmag)/255 
gradmag = (gradmag/scale_factor).astype(np.uint8) 

# Create a binary image of ones where threshold is met, zeros otherwise
binary_output = np.zeros_like(gradmag)

binary_output[(gradmag >= 30) & (gradmag <= 100)] = 1


plt.imshow(binary_output, cmap='gray')
plt.show()