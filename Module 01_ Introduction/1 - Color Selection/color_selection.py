import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print out some stats
image = mpimg.imread('/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 : Computer Vision and Deep Learning/Module 01: Introduction/1 - Color Selection/test.jpg')
print('This image is: ',type(image),'with dimensions:', image.shape)

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]

# Note: always make a copy rather than simply using "="
color_select = np.copy(image)

# Define our color selection criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# all rows all columns first channel Red " image[:,:,0] " less than threshold Red
# all rows all columns first channel Green " image[:,:,1] " less than threshold Green
# all rows all columns first channel Blue " image[:,:,2] " less than threshold Blue

thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])

# change the color of those pixels to black
color_select[thresholds] = [0,0,0]

# Display the image                 
plt.imshow(color_select)
plt.show()

# CONCLUSION : ALL THAT IS REMAINING ARE THE LANE LINES OF THE ROAD