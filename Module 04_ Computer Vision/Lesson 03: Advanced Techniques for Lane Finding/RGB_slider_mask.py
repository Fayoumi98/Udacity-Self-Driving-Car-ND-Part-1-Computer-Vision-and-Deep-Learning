#import opencv and numpy
import cv2  
import numpy as np

#trackbar callback fucntion does nothing but required for trackbar
def nothing(x):
	pass

#create a seperate window named 'controls' for trackbar
cv2.namedWindow('controls',2)
cv2.resizeWindow("controls", 550,10);

#create trackbar in 'controls' window with name 'r''
cv2.createTrackbar('r_low','controls',0,255,nothing)
cv2.createTrackbar('r_high','controls',0,255,nothing)
cv2.createTrackbar('g_low','controls',0,255,nothing)
cv2.createTrackbar('g_high','controls',0,255,nothing)
cv2.createTrackbar('b_low','controls',0,255,nothing)
cv2.createTrackbar('b_high','controls',0,255,nothing)


while(1):
	#create a black image 
	img = cv2.imread("/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 _ Computer Vision and Deep Learning/Module 04_ Computer Vision/Lesson 02: Gradients and Color Spaces/Advanced Techniques for Lane Finding/color-shadow-example.jpg")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	#returns current position/value of trackbar 
	r_low = int(cv2.getTrackbarPos('r_low','controls'))
	g_low = int(cv2.getTrackbarPos('g_low','controls'))
	b_low = int(cv2.getTrackbarPos('b_low','controls'))
	r_high = int(cv2.getTrackbarPos('r_high','controls'))
	g_high = int(cv2.getTrackbarPos('g_high','controls'))
	b_high = int(cv2.getTrackbarPos('b_high','controls'))



	
	mask_image = cv2.inRange(img, (r_low,g_low,b_low),(r_high,g_high,b_high))
	resized_image = cv2.resize(mask_image, (600, 600))
	cv2.imshow('img',resized_image)
	
	#waitfor the user to press escape and break the while loop 
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break
		
#destroys all window
cv2.destroyAllWindows()