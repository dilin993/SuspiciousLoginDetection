from __future__ import division, print_function, absolute_import
import pca
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


FEATURE_COLUMNS = ['Last Geo Velocity', 'Previous Consecutive Failures', 'Login Success',
                   'Consecutive Failure Time',
                   'IP Changed Last Time', 'No. of Failures']#, 'Maximum Geo Velocity']
LABEL_COLUMN = 'Suspicious Login'

df = pd.read_csv('features-2018-11-30-03-32-55.csv')
df = df.sample(frac=1).reset_index(drop=True)
dfLabels = df[LABEL_COLUMN].values
df = df[FEATURE_COLUMNS]

x_train, x_test, y_train, y_test = train_test_split(df, dfLabels, test_size=0.3)

model = keras.Sequential([
    keras.layers.Dense(6, input_dim=len(FEATURE_COLUMNS), activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.save('model.h5')


test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)


