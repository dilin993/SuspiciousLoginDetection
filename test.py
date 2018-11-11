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


FEATURE_COLUMNS = ['Maximum consecutive failures', 'Total failures',
                   'Time between maximum consecutive failures', 'Time between last two logins',
                   'Maximum geo-velocity', 'Geo-velocity of last login', 'Last login success']
LABEL_COLUMN = 'Suspicious login'

df = pd.read_csv('/home/dilin/wso2/projects/tf_demo/SuspiciousLoginDetection/features-2018-11-11-10-32-23.csv')
df = df.sample(frac=1).reset_index(drop=True)
labels = df[LABEL_COLUMN].values
df = df[FEATURE_COLUMNS]

rowCount = df.shape[0]

# train, test = train_test_split(df, test_size=0.1)
data = df.values

# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network Parameters
num_input = len(FEATURE_COLUMNS) # no. of features selected
num_hidden_1 = int(num_input/2) # 1st layer num features
num_hidden_2 = int(num_hidden_1/2) # 2nd layer num features (the latent dim)


# tf Graph input
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
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



# pca = pca.PCA(x)
#
# pca.fit()
#
# y = pca.reduce(keep_info=0.95)
#
# kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
#
# print float(sum(kmeans.labels_==0)) / float(len(kmeans.labels_))
#
# print df[kmeans.labels_==1]
#
# color_mapping = {0: sns.xkcd_rgb['bright purple'], 1: sns.xkcd_rgb['lime']}
# colors = list(map(lambda x: color_mapping[x], kmeans.labels_))
#
# plt.scatter(y[:, 0], y[:, 1], c=colors)
# plt.show()


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X


# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

saver = tf.train.Saver()


# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    saver.restore(sess, "./model.ckpt")
    print("Model loaded!")

    y_test = sess.run(decoder_op, feed_dict={X: data})
    meanSquareError = np.mean(np.square(data-y_test), axis=1)
    plt.hist(meanSquareError, color='blue', edgecolor='black',
             bins=int(180 / 5))
    # Add labels
    plt.title('Histogram of mean squared error')
    plt.xlabel('Mean squared error')
    plt.ylabel('Frequency')
    plt.show()

    correct = 0
    falsePositives = 0
    falseNegatives = 0
    for i in range(rowCount):
        curLabel = 0
        if meanSquareError[i] > 0.2:
            curLabel = 1
        if labels[i] == curLabel:
            correct = correct + 1
        else:
            if labels[i] == 1:
                falsePositives = falsePositives + 1
            else:
                falseNegatives = falseNegatives + 1
    print("Accuracy: ", float(correct)/float(rowCount))
    print("False positives: %d/%d"%(falsePositives, rowCount))
    print("False negatives: %d/%d" % (falseNegatives, rowCount))

