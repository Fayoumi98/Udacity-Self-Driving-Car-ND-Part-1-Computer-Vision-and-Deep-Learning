import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


img_path = '/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 _ Computer Vision and Deep Learning/Module 04_ Computer Vision/Lesson 02: Gradients and Color Spaces/Sobel Operator/curved-lane.jpg'

im = mpimg.imread(img_path)
fig , axis_array = plt.subplots(3,4)

# Individual RGB Channels
R = im[:,:,0]
G = im[:,:,1]
B = im[:,:,2]


axis_array[0,0].imshow(R,cmap='gray')
axis_array[1,0].imshow(G,cmap='gray')
axis_array[2,0].imshow(B,cmap='gray')



# 8 Bit gray threshhold 
thresh = (200, 255)


binaryR = np.zeros_like(R)
binaryG = np.zeros_like(G)
binaryB = np.zeros_like(B)

# Mask threshold pixels
binaryR[(R > thresh[0]) & (R <= thresh[1])] = 1
binaryG[(G > thresh[0]) & (G <= thresh[1])] = 1
binaryB[(B > thresh[0]) & (B <= thresh[1])] = 1



axis_array[0,1].imshow(binaryR,cmap='gray')
axis_array[1,1].imshow(binaryG,cmap='gray')
axis_array[2,1].imshow(binaryB,cmap='gray')


hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)

# Individual HLS Channels
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]


axis_array[0,2].imshow(H,cmap='gray')
axis_array[1,2].imshow(L,cmap='gray')
axis_array[2,2].imshow(S,cmap='gray')

thresh_HLS = (90, 255)

binaryH = np.zeros_like(H)
binaryL = np.zeros_like(L)
binaryS = np.zeros_like(S)


binaryH[(H > thresh_HLS[0]) & (H <= thresh_HLS[1])] = 1
binaryL[(L > thresh_HLS[0]) & (L <= thresh_HLS[1])] = 1
binaryS[(S > thresh_HLS[0]) & (S <= thresh_HLS[1])] = 1



axis_array[0,3].imshow(binaryH,cmap='gray')
axis_array[1,3].imshow(binaryL,cmap='gray')
axis_array[2,3].imshow(binaryS,cmap='gray')

plt.show()