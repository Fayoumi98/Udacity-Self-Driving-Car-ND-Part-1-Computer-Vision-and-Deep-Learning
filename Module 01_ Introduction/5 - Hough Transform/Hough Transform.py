import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in the image and convert to grayscale
# Here we read a .png and convert to 0,255 bytescale
image = mpimg.imread('/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 : Computer Vision and Deep Learning/Module 01: Introduction/5 - Hough Transform/exit-ramp.jpg')

gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# Define a triangle region of interest 
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
left_bottom = [130, 540]
right_bottom = [800, 540]
apex = [469,320]

# Define a kernel size for Gaussian smoothing / blurring
kernel_size = 5 # Must be an odd number (3, 5, 7...)
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
masked_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Define the Hough transform parameters
# Note: because the pixel space in cartesian coordinates has an infinite number of values in hough space
#       hough transform uses the polar coordinates of the pixel space and consequently represented as a sine wave in hough space
#       rho = x*cos(theta) + y*sin(theta)
# Make a blank the same size as our image to draw on
rho = 2                             # The distance of our grid in Hough space
theta = np.pi/180                   # The angular resolution of our grid in Hough space (1 degree "in radians")
threshold = 15                       # A parameter that specifies the minimum number of votes (intersections in a given grid cell)
                                    # a candidate line needs to have to make it into the output.
min_line_length = 40                # The minimum length of a line (in pixels) that you will accept in the output
max_line_gap = 20                    # The maximum distance of a line (in pixels) between segments that you will allow to be connected into a single line.
line_image = np.copy(image)*0       #creating a blank to draw lines on

# Run Hough on edge detected image
# The empty np.array([]) is just a placeholder
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((masked_edges, masked_edges, masked_edges)) 

# Draw the lines on the edge image
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
plt.imshow(combo)
plt.show()