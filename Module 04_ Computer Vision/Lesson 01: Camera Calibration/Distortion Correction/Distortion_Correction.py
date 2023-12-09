import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pickle

# Read Input image
test_image_path = '/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 _ Computer Vision and Deep Learning/Module 04_ Computer Vision/Lesson 01: Camera Calibration/Distortion Correction/calibration_wide/GOPR0033.jpg'
test = '/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 _ Computer Vision and Deep Learning/Module 04_ Computer Vision/Lesson 01: Camera Calibration/Distortion Correction/calibration_wide/GOPR0063.jpg'

test_image = mpimg.imread(test_image_path)
test = mpimg.imread(test)


objpoints = []          # 3D coordinates in world space
imgpoints = []          # 2D coordinates in image sapce


# Chess board shape 8x6x3
# Prepare object points (0,0,0),(1,0,0),(2,0,0),.....,(7,5,0)
objp = np.zeros((6*8,3),np.float32)

objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
'''
       [0., 0., 0.],[1., 0., 0.],[2., 0., 0.],[3., 0., 0.],[4., 0., 0.],[5., 0., 0.],[6., 0., 0.],[7., 0., 0.],
       [0., 1., 0.],[1., 1., 0.],[2., 1., 0.],[3., 1., 0.],[4., 1., 0.],[5., 1., 0.],[6., 1., 0.],[7., 1., 0.],
       [0., 2., 0.],[1., 2., 0.],[2., 2., 0.],[3., 2., 0.],[4., 2., 0.],[5., 2., 0.],[6., 2., 0.],[7., 2., 0.],
       [0., 3., 0.],[1., 3., 0.],[2., 3., 0.],[3., 3., 0.],[4., 3., 0.],[5., 3., 0.],[6., 3., 0.],[7., 3., 0.],
       [0., 4., 0.],[1., 4., 0.],[2., 4., 0.],[3., 4., 0.],[4., 4., 0.],[5., 4., 0.],[6., 4., 0.],[7., 4., 0.],
       [0., 5., 0.],[1., 5., 0.],[2., 5., 0.],[3., 5., 0.],[4., 5., 0.],[5., 5., 0.],[6., 5., 0.],[7., 5., 0.]],
'''

# convert image to grayscale
gray = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret , corners = cv2.findChessboardCorners(gray,(8,6),None)



if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)

    # Draw and display corners
    img = cv2.drawChessboardCorners(test_image,(8,6),corners,ret)

    # Input image shape
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    # Inputs : Object points - Image Points - Image size
    # Returns : Distorsion coefficients - Camera Matrix - Rotation vectors - Translation vectors
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    # Undistort camera image
    # Inputs : Distorsion coefficients - Camera Matrix - Image
    dst = cv2.undistort(test, mtx, dist, None, mtx)
    


    fig = plt.figure(figsize=(2,1))
    fig.add_subplot(2,1, 1)
    plt.imshow(img)
    fig.add_subplot(2,1, 2)
    plt.imshow(dst)

    plt.show()