from __future__ import division, print_function, absolute_import
from flask import Flask, request, session
from flask_restful import Resource, Api
from flask_jsonpify import jsonify
import queue
import feature_calculation
import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
from sklearn.model_selection import train_test_split

import pandas as pd
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
app = Flask(__name__)
api = Api(app)

# load the model, and pass in the custom metric function
model = keras.models.load_model('model.h5')
global graph
graph = tf.get_default_graph()


def is_suspicious(x):
    if x is None:
        return False
    with graph.as_default():
        x = (np.expand_dims(x, 0))
        y = model.predict(x)
        if np.argmax(y) == 1:
            return True


@app.route('/detection', methods=['POST'])
def publish():
    logging.info("Detection request received")
    data = request.json['event']
    logging.info("data = " + str(data))
    username = data[feature_calculation.USER_USERNAME]
    feature_calculation.insert_to_login_data(data)
    x = feature_calculation.get_features(username)
    result = {}
    result['suspicious'] = is_suspicious(x)
    logging.info("Result = " + str(result))
    return jsonify(result)


if __name__ == '__main__':
    logging.info("starting up the server...")
    app.run(debug=False)