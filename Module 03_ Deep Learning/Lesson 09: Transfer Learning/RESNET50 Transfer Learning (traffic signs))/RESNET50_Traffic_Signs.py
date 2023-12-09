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
import pickle
from keras.optimizers import Adam
from sklearn.metrics import ConfusionMatrixDisplay , confusion_matrix , accuracy_score

images = []
labels = []
classes = 43


# load German Traffic Sign Recognition Benchmark Dataset
data_path = '/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 _ Computer Vision and Deep Learning/Module 03_ Deep Learning/Lesson 09: Transfer Learning/Alexnet Trafic Signs/GTSRB/Data'



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


X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.33, random_state=42, shuffle=True)




ResNet50_pretrained_model = tf.keras.applications.ResNet50(
                include_top= False,             # to use your own input and output layers
                weights="imagenet",             # Load pretrained weights
                input_shape= (32,32,3),         # Input image shape
                pooling='avg',                  # average pooling / max pooling
                classes=43,                     # Number of classes
            )


'''
the pre-trained layers are frozen, so only the weights for the new layer(s) are trained.
In other words, the gradient doesn't flow backwards past the first new layer.
'''

'''
for layer in ResNet50_pretrained_model.layers:
    layer.trainable = False
'''

'''
for i , layer in enumerate(ResNet50_pretrained_model.layers):
    print(i,layer.name," - ",layer.trainable)
'''


ResNet50_new_model = tf.keras.Sequential()


ResNet50_new_model.add(ResNet50_pretrained_model)
# Flatten (output 1 dimension)
ResNet50_new_model.add(tf.keras.layers.Flatten())
# Fully connected layer
ResNet50_new_model.add(tf.keras.layers.Dense(1024,activation='relu'))
ResNet50_new_model.add(tf.keras.layers.Dense(512,activation='relu'))
ResNet50_new_model.add(tf.keras.layers.Dense(256,activation='relu'))
ResNet50_new_model.add(tf.keras.layers.Dense(128,activation='relu'))
ResNet50_new_model.add(tf.keras.layers.Dense(64,activation='relu'))
ResNet50_new_model.add(tf.keras.layers.Dense(43,activation='softmax'))

ResNet50_new_model.summary()


ResNet50_new_model.compile(optimizer= Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics='accuracy')

history = ResNet50_new_model.fit(X_train, y_train, batch_size=32, epochs=3,validation_data=(X_valid, y_valid))



# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()