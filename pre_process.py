import constants
import pandas as pd
import datetime
import math
import feature_calculation
import numpy as np

featureMappings = {
    constants.HEADER_USERNAME: feature_calculation.USER_USERNAME,
    constants.HEADER_TIMESTAMP: feature_calculation.USER_TIMESTAMP,
    constants.HEADER_SUSPICIOUS_LOGIN: feature_calculation.USER_SUSPICIOUS_LOGIN,
    constants.HEADER_LONGITUDE: feature_calculation.USER_LONGITUDE,
    constants.HEADER_REMOTEIP: feature_calculation.USER_REMOTEIP,
    constants.HEADER_LATITUDE: feature_calculation.USER_LATITUDE,
    constants.HEADER_COUNTRY: feature_calculation.USER_COUNTRY,
    constants.HEADER_EVENTTYPE: feature_calculation.USER_EVENTTYPE,
    constants.HEADER_AUTHENTICATIONSUCCESS: feature_calculation.USER_AUTHENTICATIONSUCCESS,
    constants.HEADER_CONTEXTID: feature_calculation.USER_CONTEXTID
}


def data_row_to_json(row):
    json = {}
    for key in featureMappings.keys():
        json[featureMappings[key]] = row[key]
    return json


def main():
    print('Reading csv...')
    df = pd.read_csv('login_data-2018-12-09-18-46-47.csv')
    features = []
    for index, row in df.iterrows():
        json = data_row_to_json(row)
        feature_calculation.insert_to_login_data(json)
        username = row[constants.HEADER_USERNAME]
        feature = feature_calculation.get_features(username, True)
        if feature is not None:
            features.append(feature)
    features = np.array(features)
    fileName = 'feature-' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.csv'
    np.savetxt(fileName, features, delimiter=',')
    print('Features saved in \'' + fileName + '\'.')


if __name__ == '__main__':
    main()
