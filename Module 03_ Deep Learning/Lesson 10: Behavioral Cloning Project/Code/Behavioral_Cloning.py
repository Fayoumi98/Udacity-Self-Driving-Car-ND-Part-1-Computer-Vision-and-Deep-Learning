import cv2
import csv
import numpy as np
import tensorflow as tf
import keras 
from keras.layers import Dense , Flatten , Lambda , Conv2D , MaxPool2D , Cropping2D
import pandas as pd
import matplotlib.pyplot as plt


csv_path = '/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 _ Computer Vision and Deep Learning/Module 03_ Deep Learning/Lesson 10: Behavioral Cloning Project/Data/driving_log.csv'

lines = []
with open(csv_path) as csvfile:
    reader =  csv.reader(csvfile)
    for line in reader:
        lines.append(line)


augmented_images = []
augmented_measurments = []

for line in lines:
    source_path = line[0]

    # array of images
    image = cv2.imread(source_path)
    augmented_images.append(image)
    augmented_images.append(cv2.flip(image,1))

    # Array of steering measurments
    measurment = float(line[3])
    augmented_measurments.append(measurment)
    augmented_measurments.append(measurment*-1)




# Keras requires numpy array format
x_train = np.array(augmented_images)
y_train = np.array(augmented_measurments)

# Regression model to predict the steering angle
model = keras.models.Sequential(layers=[
    Lambda(lambda x: x/255.0-0.5,input_shape=(160,320,3)),
    Cropping2D(cropping=((70,25),(0,0))),
    Conv2D(6,5,5,activation='relu'),
    MaxPool2D(),
    Lambda(lambda x: x/255.0-0.5,input_shape=(160,320,3)),
    MaxPool2D(),
    Flatten(),
    Dense(120),
    Dense(84),
    Dense(1),   

])


# mean squared error
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
history = model.fit(x_train,y_train,validation_split=0.2,shuffle= True,epochs=4)


# Save Model
model_path = '/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 _ Computer Vision and Deep Learning/Module 03_ Deep Learning/Lesson 10: Behavioral Cloning Project/Model/'
model.save(model_path + 'model.h5')



pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # setting limits for y-axis
plt.show()
