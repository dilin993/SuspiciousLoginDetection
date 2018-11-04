import tensorflow as tf
import numpy as np


class PCA:

    def __init__(self, data):
        self.data = data
        self.dtype = tf.float32

    def fit(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(self.dtype, shape=self.data.shape)
            # Perform SVD
            singular_values, u, _ = tf.svd(self.X)
            # Create sigma matrix
            sigma = tf.diag(singular_values)
        with tf.Session(graph=self.graph) as session:
            self.u, self.singular_values, self.sigma = session.run([u, singular_values, sigma],
                                                                   feed_dict={self.X: self.data})

    def reduce(self, n_dimensions=None, keep_info=None):
        if keep_info:
            # Normalize singular values
            normalized_singular_values = self.singular_values / sum(self.singular_values)
            # Create the aggregated ladder of kept information per dimension
            ladder = np.cumsum(normalized_singular_values)
            # Get the first index which is above the given information threshold
            index = next(idx for idx, value in enumerate(ladder) if value >= keep_info) + 1
            n_dimensions = index
        with self.graph.as_default():
            # Cut out the relevant part from sigma
            sigma = tf.slice(self.sigma, [0, 0], [self.data.shape[1], n_dimensions])
            # PCA
            pca = tf.matmul(self.u, sigma)
        with tf.Session(graph=self.graph) as session:
            return session.run(pca, feed_dict={self.X: self.data})
