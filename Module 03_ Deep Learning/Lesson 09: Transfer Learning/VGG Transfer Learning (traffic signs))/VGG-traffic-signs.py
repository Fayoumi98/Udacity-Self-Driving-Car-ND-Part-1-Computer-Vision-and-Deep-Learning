import pickle
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import keras
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam , SGD
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
# Import to_categorical from the keras utils package to one hot encode the labels
from keras.utils import to_categorical      
import cv2 as cv
import os
from PIL import Image




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






VGG_pretrained_model = tf.keras.applications.vgg16.VGG16(
                include_top= False,             # to use your own input and output layers
                weights="imagenet",             # Load pretrained weights
                input_shape= (32,32,3),         # Input image shape
                pooling='avg',                  # average pooling / max pooling
                classes=43,                     # Number of classes
            )



for layer in VGG_pretrained_model.layers:
    layer.trainable = False


new_vgg_model = tf.keras.Sequential()

new_vgg_model.add(tf.keras.layers.Flatten())
new_vgg_model.add(tf.keras.layers.Dense(128,activation='relu'))
new_vgg_model.add(tf.keras.layers.Dense(64,activation='relu'))
new_vgg_model.add(tf.keras.layers.Dense(32,activation='relu'))
new_vgg_model.add(tf.keras.layers.Dropout(0.2))
new_vgg_model.add(tf.keras.layers.Dense(43, activation='softmax'))


new_vgg_model.compile(optimizer= SGD(learning_rate=0.1),loss='sparse_categorical_crossentropy',metrics='accuracy')

history = new_vgg_model.fit(X_train, y_train, batch_size=32, epochs=12,validation_data=(X_valid, y_valid))


new_vgg_model.summary()

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