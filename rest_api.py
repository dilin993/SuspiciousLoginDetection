from __future__ import division, print_function, absolute_import
from flask import Flask, request
from flask_restful import Resource, Api
from flask_jsonpify import jsonify
import queue
import feature_calculation
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd

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
        print(x)
        y = model.predict(x)
        if np.argmax(y) == 1:
            print("Suspicious login detected !")
            return True


@app.route('/publish', methods=['POST'])
def publish():
    data = request.json['event']['payloadData']
    feature_calculation.insert_to_login_data(data)
    # x = feature_calculation.get_features(data[feature_calculation.USER_USERNAME])
    # return str(is_suspicious(x))


@app.route('/evaluate', methods=['GET'])
def evaluate():
    username = request.args.get('username')
    x = feature_calculation.get_features(username)
    return jsonify({'suspicious': is_suspicious(x)})


# class LoginAnalysis(Resource):
#
#     def post(self):
#         data = request.json['event']['payloadData']
#         feature_calculation.insert_to_login_data(data)
#         x = feature_calculation.get_features(data[feature_calculation.USER_USERNAME])
#         is_suspicious(x)
#
#     def get(self):
#         username = request.args.get('username')
#         x = feature_calculation.get_features(username)
#         return is_suspicious(x)
#
#
# api.add_resource(LoginAnalysis, '/')

if __name__ == '__main__':
    app.run(debug=False)