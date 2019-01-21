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
global userResult
userResult = {}


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
    logging.info("publish called")
    data = request.json['event']['payloadData']
    username = data[feature_calculation.USER_USERNAME]
    feature_calculation.insert_to_login_data(data)
    if username in userResult:
        userResult[username]['isReady'] = True
    x = feature_calculation.get_features(username)
    is_suspicious(x)
    return jsonify({'result': 'success'})


@app.route('/evaluate', methods=['GET'])
def evaluate():
    logging.info("evaluate called")
    username = request.args.get('username')
    if username not in userResult:
        result = {'suspicious': False, 'isReady': False, 'prevRequest': True}
        userResult[username] = result
        return jsonify(result)
    result = userResult[username]
    if not result['prevRequest']:
        result['prevRequest'] = True
        result['isReady'] = False
        userResult[username] = result
        return jsonify(result)
    if not result['isReady']:
        return jsonify(result)
    x = feature_calculation.get_features(username)
    result['suspicious'] = is_suspicious(x)
    result['prevRequest'] = False
    userResult[username] = result
    return jsonify(result)


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
    logging.info("starting up the server...")
    app.run(debug=False)