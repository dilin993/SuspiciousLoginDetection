from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

DIMENSIONS = 7
CLUSTERS = 2


class KMeansModel:

    def __init__(self):
        self.centers = tf.random_uniform(shape=(DIMENSIONS, CLUSTERS))
        self.x = tf.placeholder(tf.float32, shape=[DIMENSIONS, 1])
        self.dist = tf.matmul(self.x, self.centers, transpose_a=True)
        self.session = tf.Session()

    def predict(self, x):
