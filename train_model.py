import constants
import pandas as pd
import datetime
import math
import geolocation
import feature_calculation

def main():
    df = pd.read_csv('loginData-2018-11-22-12-00-06.csv')
    for index, row in df.iterrows():
        data = {}
        data[feature_calculation.USER_USERNAME] = str(row[constants.HEADER_USERNAME])
        data[feature_calculation.USER_CONTEXTID] = str(row[constants.HEADER_CONTEXTID])
        data[feature_calculation.USER_REMOTEIP] = str(row[constants.HEADER_REMOTEIP])
        data[feature_calculation.USER_AUTHENTICATIONSUCCESS] = str(row[constants.HEADER_AUTHENTICATIONSUCCESS])
        data[feature_calculation.USER_COUNTRY] = str(row[constants.HEADER_COUNTRY])
        data[feature_calculation.USER_EVENTTYPE] = str(row[constants.HEADER_EVENTTYPE])
        data[feature_calculation.USER_LATITUDE] = str(row[constants.HEADER_LATITUDE])
        data[feature_calculation.USER_LONGITUDE] = str(row[constants.HEADER_LONGITUDE])
        data[feature_calculation.USER_SUSPICIOUS_LOGIN] = str(row[constants.HEADER_SUSPICIOUS_LOGIN])
        data[feature_calculation.USER_TIMESTAMP] = int(row[constants.HEADER_TIMESTAMP])