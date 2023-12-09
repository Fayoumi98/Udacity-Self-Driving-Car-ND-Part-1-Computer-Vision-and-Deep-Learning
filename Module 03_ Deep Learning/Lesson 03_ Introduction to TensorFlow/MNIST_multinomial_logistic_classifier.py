import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Load train and test data (MNIST)
# Original Size Training Set : (60000, 28, 28) (60000) Test Set : (10000, 28, 28) (60000)
(x_train, y_train), (x_test, y_test)  = tf.keras.datasets.mnist.load_data()


# Converting image data to float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

# Flatten images to 1-D vector of 784 features (28*28)
# Resized Training Set : (60000, 784) (60000) Test Set : (10000, 784) (60000)
x_train, x_test = x_train.reshape(x_train.shape[0], -1), x_test.reshape(x_test.shape[0], -1)


# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# Due to the large number of images for training, it is suggested to train the images in batches
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(256).prefetch(1)



class Model:

    def __init__(self):
        # Initialize weight and bias arrays
        # Weight Matrix (784x10)
        # Bias Matrix (10x1)
        self.W = tf.Variable(tf.ones([784, 10]), name="weight")
        self.b = tf.Variable(tf.zeros([10]), name="bias")

    def __call__(self, x):
        # Logits - xW + b
        y_pred = tf.nn.softmax(tf.matmul(x, self.W) + self.b)
        return y_pred



def loss(y_pred, y_true):
    
    # Encode label to a one hot vector
    y_true = tf.one_hot(y_true, depth=10)
    
    # Clip prediction between 1e-9 , 1
    # because log(0) = error
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)

    # Compute cross-entropy
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred),1))


def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



def train(model, x, y):

    with tf.GradientTape() as t:
        pred = model(x)
        current_loss = loss(pred, y)

    # Compute gradients
    gradients = t.gradient(current_loss, [model.W, model.b])
    
    # Update W and b following gradients by the stochastic gradient descent
    optimizer.apply_gradients(zip(gradients, [model.W, model.b]))


# Initialize the model
model = Model()
epochs = 60
losses = []

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(lr = 0.1)

for epoch_count in range(epochs):

    current_loss = loss(model(x_train), y_train)
    losses.append(current_loss)

    # Train the model
    train(model, x_train, y_train)

    # Calculate Accuracy
    Accuracy = accuracy(model(x_train), y_train)
    print("Epoch No :",epoch_count+1,"Loss :",float(current_loss),"Accuracy :",float(Accuracy)*100,"%")

# Visualizing the loss function
plt.plot(losses)
plt.xlabel('Num of epochs')
plt.ylabel('Loss')
plt.show()