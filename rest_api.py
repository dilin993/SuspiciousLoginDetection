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

import pca
import pandas as pd

app = Flask(__name__)
api = Api(app)

# load the model, and pass in the custom metric function
model = keras.models.load_model('model.h5')
global graph
graph = tf.get_default_graph()


def isSuspicious(x):
    if x is None:
        return False
    with graph.as_default():
        x = (np.expand_dims(x, 0))
        print(x)
        y = model.predict(x)
        if np.argmax(y) == 1:
            print("Suspicous login detected !")

class LoginAnalysis(Resource):

    def post(self):
        data = request.json['event']['payloadData']
        feature_calculation.insertToLoginData(data)
        x = feature_calculation.getFeatures(data[feature_calculation.USER_USERNAME])
        isSuspicious(x)


api.add_resource(LoginAnalysis, '/')

if __name__ == '__main__':
    app.run(debug=False)