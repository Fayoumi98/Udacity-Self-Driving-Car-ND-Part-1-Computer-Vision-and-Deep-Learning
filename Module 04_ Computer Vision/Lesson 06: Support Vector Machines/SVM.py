import random
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score



def makeTerrainData(n_points=1000):

    ### make the toy dataset
    random.seed(42)
    grade = [random.random() for ii in range(0,n_points)]
    bumpy = [random.random() for ii in range(0,n_points)]
    error = [random.random() for ii in range(0,n_points)]
    y = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(0,n_points)]
    for ii in range(0, len(y)):
        if grade[ii]>0.8 or bumpy[ii]>0.8:
            y[ii] = 1.0

    ### split into train/test sets
    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.75*n_points)
    X_train = X[0:split]
    X_test  = X[split:]
    y_train = y[0:split]
    y_test  = y[split:]

    grade_sig = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==0]
    bumpy_sig = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==0]
    grade_bkg = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==1]
    bumpy_bkg = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==1]


    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    test_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
            , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}

    return X_train, y_train, X_test, y_test



def classify(features_train, labels_train,feature_test):   
    '''
    Intuitively, the gamma parameter defines how far the influence of a single training example reaches, 
    with low values meaning ‘far’ and high values meaning ‘close’. The gamma parameters can be seen as the 
    inverse of the radius of influence of samples selected by the model as support vectors.

    The C parameter trades off correct classification of training examples against maximization of the decision
    function’s margin. For larger values of C, a smaller margin will be accepted if the decision function is 
    better at classifying all training points correctly. A lower C will encourage a larger margin, therefore 
    a simpler decision function, at the cost of training accuracy. In other words C behaves as a regularization
    parameter in the SVM.
    '''

    '''
    For choosing C we generally choose the value like 0.001, 0.01, 0.1, 1, 10, 100
    and same for Gamma 0.001, 0.01, 0.1, 1, 10, 100

    we only tune gamma for Gaussian RBF kernel. if you use linear or polynomial kernel then you do not need gamma only you need C
    '''

    classifier = svm.SVC(kernel="rbf",gamma=1000,C=10)
    classifier.fit(X=features_train,y=labels_train)
    pred = classifier.predict(X_test)
    
    return pred
    


X_train, y_train, X_test, y_test = makeTerrainData()

predictions = classify(X_train,y_train,X_test)

accuracy = accuracy_score(y_test,predictions)

print(accuracy)