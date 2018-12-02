import constants
import pandas as pd
import datetime
import math
import geolocation

MAX_TIME = 300
MAX_GEOVELOCITY = 400

LOGIN_TABLE_COLUMNS = ['USERNAME', 'LOGIN_SUCCESS', 'TIME', 'IP', 'LONGITUDE', 'LATITUDE', 'SUSPICIOUS LOGIN',
                       'CONTEXT_ID']


# identify successful and failed logins
def preProcess(df):
    loginTableDf = pd.DataFrame(index=None, columns=LOGIN_TABLE_COLUMNS)
    i = 0
    for username in df[constants.HEADER_USERNAME].unique():
        userDf = df[df[constants.HEADER_USERNAME] == username]
        prevRow = None
        for index, row in userDf.iterrows():
            if prevRow is None:
                prevRow = row
                next
            elif prevRow[constants.HEADER_CONTEXTID] == row[constants.HEADER_CONTEXTID] and \
                    prevRow[constants.HEADER_EVENTTYPE] == constants.EVENTYPE_STEP and \
                    row[constants.HEADER_EVENTTYPE] == constants.EVENTYPE_OVERALL and row[
                constants.HEADER_AUTHENTICATIONSUCCESS] == 1:
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[0]] = username
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[1]] = 1
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[2]] = datetime.datetime.fromtimestamp(
                    row[constants.HEADER_TIMESTAMP] / 1e3)
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[3]] = row[constants.HEADER_REMOTEIP]
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[4]] = row[constants.HEADER_LONGITUDE]
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[5]] = row[constants.HEADER_LATITUDE]
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[6]] = row[constants.HEADER_SUSPICIOUS_LOGIN]
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[7]] = row[constants.HEADER_CONTEXTID]
                i = i + 1
            elif prevRow[constants.HEADER_EVENTTYPE] == constants.EVENTYPE_STEP and prevRow[
                constants.HEADER_AUTHENTICATIONSUCCESS] == 0:
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[0]] = username
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[1]] = 0
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[2]] = datetime.datetime.fromtimestamp(
                    prevRow[constants.HEADER_TIMESTAMP] / 1e3)
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[3]] = prevRow[constants.HEADER_REMOTEIP]
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[4]] = prevRow[constants.HEADER_LONGITUDE]
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[5]] = prevRow[constants.HEADER_LATITUDE]
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[6]] = prevRow[constants.HEADER_SUSPICIOUS_LOGIN]
                loginTableDf.at[i, LOGIN_TABLE_COLUMNS[7]] = prevRow[constants.HEADER_CONTEXTID]
                i = i + 1
            prevRow = row
        if prevRow[constants.HEADER_EVENTTYPE] == constants.EVENTYPE_STEP and prevRow[
            constants.HEADER_AUTHENTICATIONSUCCESS] == 0:
            loginTableDf.at[i, LOGIN_TABLE_COLUMNS[0]] = username
            loginTableDf.at[i, LOGIN_TABLE_COLUMNS[1]] = 0
            loginTableDf.at[i, LOGIN_TABLE_COLUMNS[2]] = datetime.datetime.fromtimestamp(
                prevRow[constants.HEADER_TIMESTAMP] / 1e3)
            loginTableDf.at[i, LOGIN_TABLE_COLUMNS[3]] = prevRow[constants.HEADER_REMOTEIP]
            loginTableDf.at[i, LOGIN_TABLE_COLUMNS[4]] = prevRow[constants.HEADER_LONGITUDE]
            loginTableDf.at[i, LOGIN_TABLE_COLUMNS[5]] = prevRow[constants.HEADER_LATITUDE]
            loginTableDf.at[i, LOGIN_TABLE_COLUMNS[6]] = prevRow[constants.HEADER_SUSPICIOUS_LOGIN]
            loginTableDf.at[i, LOGIN_TABLE_COLUMNS[7]] = prevRow[constants.HEADER_CONTEXTID]
            i = i + 1
    return loginTableDf


def getTimeDiff(timestamp1, timestamp2):
    diff = abs((timestamp1 - timestamp2).seconds)
    return diff


def getFeatures(loginTableDf, windowSize):
    j = 0
    WINDOW_SIZE = windowSize
    FEATURE_COLUMNS = ['Last Geo Velocity', 'Previous Consecutive Failures', 'Login Success',
                       'Consecutive Failure Time', 'IP Changed Last Time', 'No. of Failures', 'Maximum Geo Velocity',
                       'Suspicious Login']
    featureDf = pd.DataFrame(index=None, columns=FEATURE_COLUMNS)
    for username in loginTableDf[LOGIN_TABLE_COLUMNS[0]].unique():
        userDf = loginTableDf[loginTableDf[LOGIN_TABLE_COLUMNS[0]] == username]
        prevRows = []
        for index, row in userDf.iterrows():
            if len(prevRows) < WINDOW_SIZE:
                prevRows.append(row)
                continue
            suspiciousLogin = row[LOGIN_TABLE_COLUMNS[6]]


            prevConsecutiveFailures = 0
            consecutiveFailureTime = 0.0
            loginSucess = prevRows[WINDOW_SIZE - 1][LOGIN_TABLE_COLUMNS[1]]
            lastGeoVelocity = 0.0
            maxGeoVelocity = 0
            numFailures = 0
            ipChangedLastTime = 0

            curContextID = prevRows[WINDOW_SIZE - 1][LOGIN_TABLE_COLUMNS[7]]
            curTime = prevRows[WINDOW_SIZE - 1][LOGIN_TABLE_COLUMNS[2]]
            lastTimeDiff = getTimeDiff(prevRows[WINDOW_SIZE - 1][LOGIN_TABLE_COLUMNS[2]],
                                       prevRows[WINDOW_SIZE - 2][LOGIN_TABLE_COLUMNS[2]])
            lastDist = abs(geolocation.calculateDistance(prevRows[WINDOW_SIZE - 1][LOGIN_TABLE_COLUMNS[4]],
                                                prevRows[WINDOW_SIZE - 1][LOGIN_TABLE_COLUMNS[5]],
                                                prevRows[WINDOW_SIZE - 2][LOGIN_TABLE_COLUMNS[4]],
                                                prevRows[WINDOW_SIZE - 2][LOGIN_TABLE_COLUMNS[5]]))

            if lastDist != 0:
                lastGeoVelocity = lastDist / lastTimeDiff

            if prevRows[WINDOW_SIZE - 1][LOGIN_TABLE_COLUMNS[3]] != prevRows[WINDOW_SIZE - 2][LOGIN_TABLE_COLUMNS[3]]:
                ipChangedLastTime = 1

            for i in range(WINDOW_SIZE-2,-1,-1):
                if prevRows[i][LOGIN_TABLE_COLUMNS[7]] == curContextID:
                    prevConsecutiveFailures = prevConsecutiveFailures + 1
                    consecutiveFailureTime = getTimeDiff(curTime,  prevRows[i][LOGIN_TABLE_COLUMNS[2]])
                else:
                    break

            for i in range(WINDOW_SIZE):
                if i > 0:
                    dt = getTimeDiff(prevRows[i - 1][LOGIN_TABLE_COLUMNS[2]], prevRows[i][LOGIN_TABLE_COLUMNS[2]])
                    dist = geolocation.calculateDistance(prevRows[i - 1][LOGIN_TABLE_COLUMNS[4]],
                                                prevRows[i - 1][LOGIN_TABLE_COLUMNS[5]],
                                                prevRows[i][LOGIN_TABLE_COLUMNS[4]],
                                                prevRows[i][LOGIN_TABLE_COLUMNS[5]])
                    geoVelocity = 0
                    if dist != 0 and dt != 0:
                        geoVelocity = dist / dt
                    if maxGeoVelocity < geoVelocity:
                        maxGeoVelocity = geoVelocity

                if prevRows[i][LOGIN_TABLE_COLUMNS[1]] == 0:
                    numFailures = numFailures + 1
            del prevRows[0]

            # add new feature row
            featureDf.at[j, FEATURE_COLUMNS[0]] = float(min(lastGeoVelocity, MAX_GEOVELOCITY)) / float(MAX_GEOVELOCITY)
            featureDf.at[j, FEATURE_COLUMNS[1]] = float(prevConsecutiveFailures) / float(WINDOW_SIZE)
            featureDf.at[j, FEATURE_COLUMNS[2]] = float(loginSucess)
            featureDf.at[j, FEATURE_COLUMNS[3]] = float(min(consecutiveFailureTime, MAX_TIME)) / float(MAX_TIME)
            featureDf.at[j, FEATURE_COLUMNS[4]] = float(min(maxGeoVelocity, MAX_GEOVELOCITY)) / float(MAX_GEOVELOCITY)
            featureDf.at[j, FEATURE_COLUMNS[5]] = float(ipChangedLastTime)
            featureDf.at[j, FEATURE_COLUMNS[6]] = float(numFailures) / float(WINDOW_SIZE)
            featureDf.at[j, FEATURE_COLUMNS[7]] = float(suspiciousLogin)
            j = j + 1
    return featureDf

def main():
    print('Reading csv...')
    df = pd.read_csv('loginData-2018-11-30-01-24-51.csv')
    print('Pre processing data...')
    loginTableDf = preProcess(df)
    fileName = 'loginProcessed-' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.csv'
    print('Saving processed logins to\'' + fileName + '\'')
    loginTableDf.to_csv(fileName)
    print('Calculating features...')
    featureDf = getFeatures(loginTableDf, 5)
    fileName = 'features-' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.csv'
    print('Saving features to\'' + fileName + '\'')
    featureDf.to_csv(fileName)


if __name__ == '__main__':
    main()
