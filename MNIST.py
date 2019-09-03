# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:02:32 2019

@author: jered.willoughby
"""

#Load libraries
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

mnist = fetch_openml('mnist_784',version=1)
mnist.keys()
#Descriptive statistics
X,y = mnist["data"], mnist["target"]
X.shape
y.shape

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)
#Select 0 indexed image from the mnist data and plot it
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()
#validate from the target set
y[0] #this is a string value
#Change datatype to integer
y = y.astype(np.uint8)
#Create the testing and training sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#Now we need to determine what classifier to use. Let's start with a binary
#target vector classification
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#Binary classifier = stochastic gradient descent SGDClassifier
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)

sgd_clf.predict([some_digit])

#Cross Validation
skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
    
#Determine cross validation score - 3 folds
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
#Prediction set selection from cross val
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
#Confusion matrix from prior variable set
confusion_matrix(y_train_5, y_train_pred)
#Performance metric - precision + recall score
precision_score(y_train_5, y_train_pred)

recall_score(y_train_5, y_train_pred)
#We can combine the two into the F1 score - harmonic mean
f1_score(y_train_5, y_train_pred)


#Note that there is a way to get the optimal threshold: decision_function() 
#method, which returns a score for each instance, and then use any threshold 
#you want to make predictions based on those scores.