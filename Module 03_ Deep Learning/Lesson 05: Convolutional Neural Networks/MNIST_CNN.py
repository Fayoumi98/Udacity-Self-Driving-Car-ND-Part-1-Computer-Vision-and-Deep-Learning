import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load train and test data (MNIST)
# Original Size Training Set : (60000, 28, 28) (60000) Test Set : (10000, 28, 28) (60000)
(x_train, y_train), (x_test, y_test)  = tf.keras.datasets.mnist.load_data()


# Check for null values in training and test sets
nan_train = np.isnan(x_train).any()
nan_test = np.isnan(x_test).any()
'''TEST : print(nan_test,nan_train)'''


# Normalization and Reshaping
input_shape = (28, 28, 1)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train=x_train / 255.0

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test=x_test/255.0

# One hot encoding 
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

# To visualize an image in the dataset
#plt.imshow(x_train[100])
#print("One Hot Encoded Labels : ",y_train[100])
#plt.show()


batch_size = 64
num_classes = 10
epochs = 5

# The model contains various layers stacked on top of each other
model = tf.keras.models.Sequential([

    # Conv2D is a convolution layer with 32 filters(kernels) (output depth k = 32)
    # Kernel = 5x5      Stride = (1,1)      Input Shape (28,28,1)       (same padding) P = 0
    # Output Width = ((W-F+2P)/s)+1
    # Output Width = ((H-F+2P)/s)+1
    # Output Depth  = 32
    # relu is the rectifier, and it is used to find nonlinearity in the data. It works by returning 
    # the input value if the input value >= 0. If the input is negative, it returns 0.
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=input_shape),
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
    
    # MaxPool2D is a downsampling filter. It reduces a 2x2 matrix of the image to a single pixel 
    # with the maximum value of the 2x2 matrix. The filter aims to conserve the main features of 
    # the image while reducing the size.
    tf.keras.layers.MaxPool2D(),
    
    # Dropout is a regularization layer. In our model, 25% of the nodes in the layer are randomly
    # ignores, allowing the network to learn different features. This prevents overfitting.
    tf.keras.layers.Dropout(0.25),


    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),

    # Max Pooling layer
    tf.keras.layers.MaxPool2D(strides=(2,2)),

    # Dropout is a regularization layer. In our model, 25% of the nodes in the layer are randomly 
    # ignores, allowing the network to learn different features. This prevents overfitting.
    tf.keras.layers.Dropout(0.25),
    
    # Flatten converts the tensors into a 1D vector.
    tf.keras.layers.Flatten(),

    # The Dense layers are an artificial neural network (ANN). The last layer returns 
    # the probability that an image is in each class (one for each digit).
    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])



model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])



# The next step is to fit our training data. If we achieve a certain level of accuracy, it may not be 
# necessary to continue training the model, especially if time and resources are limited.he following 
# cell defines a CallBack so that if 99.5% accuracy is achieved, the model stops training. The model 
# is not likely to stop prematurely if only 5 epochs are specified. 
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.995):
      print("\nReached 99.5% accuracy so cancelling training!")
      self.model.stop_training = True



callbacks = myCallback()


# Testing the model on a validation dataset prevents overfitting of the data. We specified a 
# 10% validation and 90% training split.
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=[callbacks])




# The accuracy increases over time and the loss decreases over time. However, the accuracy of our 
# validation set seems to slightly decrease towards the end even thought our training accuracy 
# increased. Running the model for more epochs might cause our model to be susceptible to overfitting.
ig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training Loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True)



test_loss, test_acc = model.evaluate(x_test, y_test)



# Predict the values from the testing dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert testing observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1)
# compute the confusion matrix
confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes) 


plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g')

plt.show()

# CONCLUSION 
# There seems to be a slightly higher confusion between (0,6) and (4,9). This is reasonable as 0's 
# and 6's look similar with their loops and 4's and 9's can be mistaken when the 4's are more rounded 
# and 9's are more angular.