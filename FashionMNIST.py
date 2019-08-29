# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:42:55 2019

@author: jered.willoughby
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

#Load the Fashion MNIST dataset from keras
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

#When loading MNIST or Fashion MNIST using Keras rather than Scikit-Learn, 
#one important difference is that every image is represented as a 28 × 28 array 
#rather than a 1D array of size 784. Moreover, the pixel intensities are 
#represented as integers (from 0 to 255) rather than floats (from 0.0 to 255.0).
X_train_full.shape
#(60000, 28, 28)
X_train_full.dtype
#dtype('uint8')

#Create validation set and scale the pixel intesities for gradient descent
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

#Assign Class names - Fashion
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#Neural Network build - sequential API = single stack of layers connected sequentially
model = keras.models.Sequential()
#First layer and add it to the model. It is a Flatten layer whose role is to 
#convert each input image into a 1D array
model.add(keras.layers.Flatten(input_shape=[28, 28]))
#Dense hidden layer with 300 neurons
model.add(keras.layers.Dense(300, activation="relu"))
#second Dense hidden layer with 100 neurons, also using the ReLU activation function
model.add(keras.layers.Dense(100, activation="relu"))
#Dense output layer with 10 neurons (one per class), using the softmax 
#activation function (because the classes are exclusive)
model.add(keras.layers.Dense(10, activation="softmax"))
#model summary
model.summary()
#Parameters
hidden1 = model.layers[1]
weights, biases = hidden1.get_weights()
weights
weights.shape
biases
biases.shape
#Compile the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
#Train the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
#Evaluate the Model
model.evaluate(X_test, y_test)

#Prediction
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

y_pred = model.predict_classes(X_new)
y_pred

np.array(class_names)[y_pred]
y_new = y_test[:3]
y_new