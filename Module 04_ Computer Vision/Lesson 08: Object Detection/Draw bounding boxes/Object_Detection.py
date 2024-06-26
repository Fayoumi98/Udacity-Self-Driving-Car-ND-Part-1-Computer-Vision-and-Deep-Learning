import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 _ Computer Vision and Deep Learning/Module 04_ Computer Vision/Lesson 08: Object Detection/draw bounding boxes/bbox-example-image.jpg')

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):

    # Make a copy of the image
    draw_img = np.copy(img)

    # Iterate through the bounding boxes
    for bbox in bboxes:
        
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)

    # Return the image copy with boxes drawn
    return draw_img

# Here are the bounding boxes I used
bboxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)), 
          ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]


result = draw_boxes(image, bboxes)


plt.imshow(result)
plt.show()