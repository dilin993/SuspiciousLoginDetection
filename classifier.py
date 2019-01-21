from __future__ import division, print_function, absolute_import
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
from keras.utils import plot_model

data = np.genfromtxt('feature-2018-12-09-21-36-56.csv', delimiter=',')
N = data[0].size - 1

x_train, x_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:, -1], test_size=0.1)

graph = tf.get_default_graph()

model = keras.Sequential([
    keras.layers.Dense(6, input_dim=N, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

with graph.as_default():
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    logdir = '../logdir'
    writer = tf.summary.FileWriter(logdir=logdir, graph=graph)
    writer.flush()
    print(' Logs saved in \'' + logdir + '\'. Run the following command to visualize the model in tensorboard:')
    print(' tensorboard --logdir ' + logdir)

model.fit(x_train, y_train, epochs=5)  # , callbacks=callbacks_list)

model.save('model.h5')

test_loss, test_acc = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred, target_names=['Non-suspicious', 'Suspicious']))

print('Test accuracy:', test_acc)
plot_model(model, to_file='../model.png', show_shapes=True)
