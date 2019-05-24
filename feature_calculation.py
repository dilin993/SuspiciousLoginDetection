import queue
import numpy as np
import datetime
import geolocation
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

DATA_USERNAME = 'username'
DATA_TIMESTAMP = 'timestamp'
DATA_START_TIMESTAMP = 'startTimestamp'
DATA_REMOTE_IP = 'remoteIp'
DATA_LOGIN_SUCCESS = 'loginSuccess'
DATA_FAILURE_COUNT = 'failureCount'

WINDOW_SIZE = 5

geoLocationForIP = {}
userData = []

FEATURE_USERNAME = 'username'
FEATURE_WEEKDAY = 'weekday'
FEATURE_HOUR = 'hour'
# FEATURE_LOGIN_SUCCESS = 'loginSuccess'
FEATURE_FAILURE_COUNT = 'failureCount'
FEATURE_LONGITUDE = 'longitude'
FEATURE_LATITUDE = 'latitude'
FEATURE_LOGIN_TIME = 'loginTime'

BATCH_COUNT = 10


class Data:
    username = ""
    timestamp = None
    startTimestamp = None
    remoteIp = ""
    loginSuccess = 0
    failureCount = 0
    longitude = 0.0
    latitude = 0.0

    def __init__(self, json):
        if DATA_USERNAME in json:
            self.username = json[DATA_USERNAME]
        if DATA_TIMESTAMP in json:
            self.timestamp = datetime.datetime.fromtimestamp(json[DATA_TIMESTAMP] / 1e3)
        if DATA_START_TIMESTAMP in json:
            self.startTimestamp = datetime.datetime.fromtimestamp(json[DATA_START_TIMESTAMP] / 1e3)
        if DATA_REMOTE_IP in json:
            self.remoteIp = json[DATA_REMOTE_IP]
            location = None
            if self.remoteIp not in geoLocationForIP:
                location = geolocation.get_geolocation(self.remoteIp)
                geoLocationForIP[self.remoteIp] = location
            else:
                location = geoLocationForIP[self.remoteIp]
            self.longitude = location.longitude
            self.latitude = location.latitude

        if DATA_LOGIN_SUCCESS in json:
            self.loginSuccess = json[DATA_LOGIN_SUCCESS]
        if DATA_FAILURE_COUNT in json:
            self.failureCount = json[DATA_FAILURE_COUNT]

    def get_features(self):
        features = dict()
        features[FEATURE_USERNAME] = self.username
        features[FEATURE_LOGIN_TIME] = (self.timestamp - self.startTimestamp).seconds
        features[FEATURE_FAILURE_COUNT] = self.failureCount
        features[FEATURE_LONGITUDE] = self.longitude
        features[FEATURE_LATITUDE] = self.latitude
        features[FEATURE_WEEKDAY] = self.timestamp.weekday()
        features[FEATURE_HOUR] = self.timestamp.hour
        return features


def collect_features(json):
    global userData
    data = Data(json)
    userData.append(data.get_features())
    logging.info('Collected data ' + str(len(userData)) + ' of ' + str(BATCH_COUNT))
    if len(userData) >= BATCH_COUNT:
        df = pd.DataFrame(userData)
        filename = 'login_data.csv'
        df.to_csv(filename, index=False, mode='a')
        logging.info('Login data saved in \'' + filename + '\'.')
        userData = []





