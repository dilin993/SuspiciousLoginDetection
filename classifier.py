from __future__ import division, print_function, absolute_import
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report


data = np.genfromtxt('feature-2018-12-09-21-36-56.csv', delimiter=',')
N = data[0].size - 1

x_train, x_test, y_train, y_test = train_test_split(data[:,0:-1], data[:,-1], test_size=0.3)

model = keras.Sequential([
    keras.layers.Dense(6, input_dim=N, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.save('model.h5')


test_loss, test_acc = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred, target_names=['Non-suspicious', 'Suspicious']))


print('Test accuracy:', test_acc)


