import tensorflow as tf
import numpy as np
import keras
from matplotlib import pyplot as plt
import seaborn as sns
import random
import os
import cv2 as cv
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
from scikitplot.metrics import plot_confusion_matrix


images = []
labels = []
classes = 43


# load German Traffic Sign Recognition Benchmark Dataset
data_path = '/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 _ Computer Vision and Deep Learning/Module 03_ Deep Learning/Lesson 07: LeNet for Traffic Signs/LeNET_traffic_signs/GTSRB/Data'



for i in range(classes):

    img_folder = data_path + '/' + str(i) + '/'

    for image_path in os.listdir(img_folder):
        try:
            image = cv.imread(os.path.join(img_folder,image_path))
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((32, 32))
            images.append(np.array(resize_image))
            labels.append(i)
        except:
            pass
    
images = np.array(images)
images = images/255
labels = np.array(labels)


X = images.astype(np.float32)
Y = labels.astype(np.float32)


X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

X_train = X_train/255 
X_valid = X_valid/255

print("X_train.shape", X_train.shape)
print("X_valid.shape", X_valid.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_valid.shape)


#Converting the labels into one hot encoding
y_train = keras.utils.to_categorical(y_train, classes)
y_valid = keras.utils.to_categorical(y_valid, classes)




model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), strides=1, activation='tanh', input_shape=(32,32,3)),
    tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=1, activation='tanh'),
    tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=2),
    tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), activation='tanh'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=84, activation='tanh'),
    tf.keras.layers.Dense(units=43, activation='softmax'),
])

model.summary()


#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


epochs = 12
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,validation_data=(X_valid, y_valid))



pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # setting limits for y-axis
plt.show()
