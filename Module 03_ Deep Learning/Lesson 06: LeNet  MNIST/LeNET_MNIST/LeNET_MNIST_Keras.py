import tensorflow as tf
import numpy as np
import keras
from matplotlib import pyplot as plt
import seaborn as sns


(train_x, train_y), (x_test, y_test) = keras.datasets.mnist.load_data()

train_x = train_x / 255.0
x_test = x_test / 255.0

train_x = tf.expand_dims(train_x, 3)
x_test = tf.expand_dims(x_test, 3)

val_x = train_x[:5000]
val_y = train_y[:5000]


print("Training Set:   {} samples".format(len(train_x)))
print("Validation Set: {} samples".format(len(val_x)))
print("Test Set:       {} samples".format(len(x_test)))


lenet_5_model = tf.keras.models.Sequential([
    # Convolution - Activation Function (tanh)
    tf.keras.layers.Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=train_x[0].shape, padding='same'), 
    # C1 : feature maps 6 @ 28x28

    # Average Pooling (Subsampling)
    tf.keras.layers.AveragePooling2D(),
    # S2 : feature maps 6 @ 14x14

    # Convolution - Activation Function (tanh)
    tf.keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'),
    # C3 : feature maps 16 @ 10x10

    # Average Pooling (Subsampling)
    tf.keras.layers.AveragePooling2D(),
    # S4 : feature maps 16 @ 5x5

    # Convolution - Activation Function (tanh)
    tf.keras.layers.Conv2D(120, kernel_size=5, strides=1, activation='tanh', padding='valid'),
    # C5 : layer 120
    
    # Flatten (to make the multidimensional input one-dimensional, commonly used in the transition from the convolution layer to the full connected layer)
    tf.keras.layers.Flatten(),  

    # Fully Connected Layer
    tf.keras.layers.Dense(84, activation='tanh'), 
    # F6 : 84

    # Output layer
    # Fully Connected Layer
    tf.keras.layers.Dense(10, activation='softmax') 
    # Output : 10
])



lenet_5_model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['acc'])

lenet_5_model.fit(train_x, train_y, epochs=5, validation_data=(val_x, val_y))

test_loss, test_acc = lenet_5_model.evaluate(x_test, y_test)


print(test_loss, test_acc)