import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2




def warp(img):

    image_size = (img.shape[1] , img.shape[0])

    # Four Source Coordinates
    src = np.float32([
        [1052,395],
        [232,552],
        [1112,882],
        [208,809]
    ])

    # Four Desired Coordinates
    dst = np.float32([
        [1052,395],
        [230,395],
        [1052,882],
        [230,882]
    ])

    # Compute the prespective transform, M
    M = cv2.getPerspectiveTransform(src,dst)

    # Could compute the inverse also by swaping the input parameters
    Minv  = cv2.getPerspectiveTransform(dst,src)

    # Create transformed image - uses linear interpolation
    warped = cv2.warpPerspective(img,M,image_size,flags=cv2.INTER_LINEAR) 

    return warped





img_path = '/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 _ Computer Vision and Deep Learning/Module 04_ Computer Vision/Lesson 01: Camera Calibration/Prespective Transform/stop_sign.jpg'
img = plt.imread(img_path)

warped_image = warp(img)


fig = plt.figure(figsize=(2,1))
fig.add_subplot(2,1, 1)
plt.imshow(warped_image)

fig.add_subplot(2,1, 2)
plt.imshow(img)
plt.plot(1052,395,'.')          # Top Right
plt.plot(232,552,'.')           # Top Left
plt.plot(1112,882,'.')          # Bottom Right
plt.plot(208,809,'.')           # Bottom Left
plt.show()



