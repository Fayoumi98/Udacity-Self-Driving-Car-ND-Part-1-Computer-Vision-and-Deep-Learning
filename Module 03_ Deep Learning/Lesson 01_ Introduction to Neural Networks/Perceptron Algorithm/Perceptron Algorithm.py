import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

    
def perceptronStep(X,Y,W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)

        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
        
    return W, b

def trainPerceptronAlgorithm(X,Y,learn_rate = 0.01, num_epochs = 25):
        
    # Initialize random weights and bias
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0]

    # Run the preceptron before tuning weights
    y_hat_predicted = validate_prediction(X=x,W=W,b=b)
    #print("Results before training:")
    #print(y_hat_predicted)

    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X,Y, W, b, learn_rate)
        
    return W , b




def validate_prediction(X,W,b):
    y_hat_predicted = []

    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        y_hat_predicted.append(y_hat)

    return y_hat_predicted


x = []
data = pd.read_csv('/home/fayo/Udacity - Self Driving Car Nanodegree/Part 1 : Computer Vision and Deep Learning/Module 03: Deep Learning/Lesson 01: Introduction to Neural Networks/Perceptron Algorithm/data.csv')

x1 = data['x1'].tolist()
x2 = data['x2'].tolist()
y = data['label'].tolist()

for i in range(len(x1)):
    x.append([x1[i],x2[i]])


W , b = trainPerceptronAlgorithm(X=x,Y=y)
y_hat_predicted = validate_prediction(X=x,W=W,b=b)

x1 = -b/W[0]
x2 = -b/W[1]

line_eq_x = [x1[0],0]
line_eq_y = [0,x2[0]]


plt.plot(line_eq_x,line_eq_y,Color='black')
print("Results after training:")
print(y_hat_predicted)
print("The true values are :")
print(y)


for i in range(len(x)):
    if y[i]==1:
        plt.scatter(x[i][0],x[i][1],Color='red')
    else:
        plt.scatter(x[i][0],x[i][1],Color='blue')

plt.show()