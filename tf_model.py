from __future__ import division, print_function, absolute_import
import pca
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np


class TensorflowModel:

    def __init__(self):
        # Training Parameters
        self.learning_rate = 0.01
        self.num_steps = 30000
        self.batch_size = 256

        # Network Parameters
        self.num_input = 7  # no. of features selected
        self.num_hidden_1 = 4  # 1st layer num features
        self.num_hidden_2 = 2  # 2nd layer num features (the latent dim)

        display_step = 1000
        examples_to_show = 10

        # tf Graph input
        self.X = tf.placeholder("float", [None, self.num_input])

        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input])),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.num_input])),
        }

        # Building the encoder
        def encoder(x):
            # Encoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                           biases['encoder_b1']))
            # Encoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                           biases['encoder_b2']))
            return layer_2

        # Building the decoder
        def decoder(x):
            # Decoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                           biases['decoder_b1']))
            # Decoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                           biases['decoder_b2']))
            return layer_2

        # Construct model
        self.encoder_op = encoder(X)
        self.decoder_op = decoder(self.encoder_op)

        # Prediction
        self.y_pred = self.decoder_op
        # Targets (Labels) are the input data.
        self.y_true = self.X

        # Define loss and optimizer, minimize the squared error
        self.loss = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

    def train(self, x, saveModelPath=None):
        # Start Training
        # Start a new TF session
        with tf.Session() as sess:

            # Run the initializer
            sess.run(self.init)

            # Training
            for i in range(1, self.num_steps + 1):

                # Run optimization op (backprop) and cost op (to get loss value)
                _, l = sess.run([self.optimizer, self.loss], feed_dict={self.X: x})
                # Display logs per step
                if i % self.display_step == 0 or i == 1:
                    print('Step %i: Minibatch Loss: %f' % (i, l))
            if saveModelPath is None:
                saveModelPath = './model.ckpt'
            save_path = self.saver.save(sess, saveModelPath)
            print('Model saved in path: %s' % save_path)

    def restore(self, savedModelPath):
        with tf.Session() as sess:
            self.saver.restore(sess, "./model.ckpt")

    def predict(self, x):
        with tf.Session() as sess:
            y = sess.run(self.decoder_op, feed_dict={self.X: x})
            return y

