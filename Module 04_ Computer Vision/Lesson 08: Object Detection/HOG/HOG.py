import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog




def data_look(car_list, notcar_list):

    data_dict = {}

    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)

    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)

    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])

    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape

    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype

    # Return data_dict
    return data_dict



def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True,
                     feature_vec=True):
                         
    """
    Function accepts params and returns HOG features (optionally flattened) and an optional matrix for 
    visualization. Features will always be the first return (flattened if feature_vector= True).
    A visualization matrix will be the second return if visualize = True.
    """
    
    return_list = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys', transform_sqrt=False, 
                                  feature_vector= feature_vec,visualize=True)
    
    # name returns explicitly
    hog_features = return_list[0]

    if vis:
        hog_image = return_list[1]
        return hog_features, hog_image
    else:
        return hog_features




non_cars_images = glob.glob('/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 _ Computer Vision and Deep Learning/Module 04_ Computer Vision/Lesson 08: Object Detection/HOG/non-vehicles_smallset/non-vehicles_smallset/notcars1/*.jpeg')
cars_images = glob.glob('/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 _ Computer Vision and Deep Learning/Module 04_ Computer Vision/Lesson 08: Object Detection/HOG/vehicles_smallset/vehicles_smallset/cars1/*.jpeg')

cars = []
notcars = []


for image in cars_images:
        cars.append(image)
        
for image in non_cars_images:
        notcars.append(image)



    
data_info = data_look(cars, notcars)

print('Your function returned a count of', 
      data_info["n_cars"], ' cars and', 
      data_info["n_notcars"], ' non-cars')

print('of size: ',data_info["image_shape"], ' and data type:', 
      data_info["data_type"])


# Just for fun choose random car / not-car indices and plot example images   
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))
    

# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])
gray = cv2.cvtColor(car_image, cv2.COLOR_RGB2GRAY)


# Call our function with vis=True to see an image output
features, hog_image = get_hog_features(gray, orient= 9, 
                        pix_per_cell= 8, cell_per_block= 2, 
                        vis=True, feature_vec=False)




# Plot the examples
fig1 = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')

fig2 = plt.figure()
plt.subplot(121)
plt.imshow(gray,cmap='gray')
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(hog_image,cmap='gray')
plt.title('HOG Visualization')

plt.show()