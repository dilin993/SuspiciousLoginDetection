import constants
import pandas as pd
import datetime
import math
import geolocation

df = pd.read_csv('loginData-2018-11-18-10-35-59.csv')

LOGIN_TABLE_COLUMNS = ['USERNAME', 'LOGIN_SUCCESS', 'TIME', 'IP', 'LONGITUDE', 'LATITUDE', 'SUSPICIOUS LOGIN']
loginTableDf = pd.DataFrame(index=None, columns=LOGIN_TABLE_COLUMNS)

## identify succesful and failed logins
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
        i = i + 1

print loginTableDf

MAX_DURATION = 300  # seconds


def getTimeDiff(timestamp1, timestamp2):
    diff = abs((timestamp1 - timestamp2).seconds)
    return diff


j = 0
WINDOW_SIZE = 5
FEATURE_COLUMNS = ['Maximum consecutive failures', 'Total failures', 'Time between maximum consecutive failures',
                   'Time between last two logins', 'Maximum geo-velocity', 'Geo-velocity of last login',
                   'Last login success', 'Suspicious login']
featureDf = pd.DataFrame(index=None, columns=FEATURE_COLUMNS)
for username in loginTableDf[LOGIN_TABLE_COLUMNS[0]].unique():
    userDf = loginTableDf[loginTableDf[LOGIN_TABLE_COLUMNS[0]] == username]
    prevRows = []
    for index, row in userDf.iterrows():
        if len(prevRows) < WINDOW_SIZE:
            prevRows.append(row)
            continue
        suspiciousLogin = row[LOGIN_TABLE_COLUMNS[6]]
        maxConsecutiveFailures = 0
        totalFailures = 0
        timeBetweenMaxCosecutiveFailures = MAX_DURATION
        timeBetweenLastTwoLogins = MAX_DURATION
        maxGeoVelocity = 0
        geoVelocityForLastLogin = 0
        lastLoginStatus = 0  # 1 for success, 0 for failure
        consecutiveFailures = 0
        consecutiveFailureStartTime = None
        lastLoginTime = None
        for i in range(WINDOW_SIZE):
            geoVelocity = 0
            if i > 0:
                dt = getTimeDiff(prevRows[i - 1][LOGIN_TABLE_COLUMNS[2]], prevRows[i][LOGIN_TABLE_COLUMNS[2]])
                dist = geolocation.distance(prevRows[i - 1][LOGIN_TABLE_COLUMNS[4]],
                                            prevRows[i - 1][LOGIN_TABLE_COLUMNS[5]],
                                            prevRows[i][LOGIN_TABLE_COLUMNS[4]], prevRows[i][LOGIN_TABLE_COLUMNS[5]])
                geoVelocity = 0
                if dist != 0 and dt != 0:
                    geoVelocity = dist / dt
                if maxGeoVelocity < geoVelocity:
                    maxGeoVelocity = geoVelocity

            if prevRows[i][LOGIN_TABLE_COLUMNS[1]] == 0:
                consecutiveFailures = consecutiveFailures + 1
                totalFailures = totalFailures + 1
                if consecutiveFailureStartTime is None:
                    consecutiveFailureStartTime = prevRows[i][LOGIN_TABLE_COLUMNS[2]]
                if consecutiveFailures > maxConsecutiveFailures:
                    maxConsecutiveFailures = consecutiveFailures
                    timeBetweenMaxCosecutiveFailures = getTimeDiff(prevRows[i][LOGIN_TABLE_COLUMNS[2]],
                                                                   consecutiveFailureStartTime)
            else:
                consecutiveFailures = 0
                consecutiveFailureStartTime = None
                if lastLoginTime is None:
                    lastLoginTime = prevRows[i][LOGIN_TABLE_COLUMNS[2]]
                timeBetweenLastTwoLogins = getTimeDiff(prevRows[i][LOGIN_TABLE_COLUMNS[2]], lastLoginTime)
                geoVelocityForLastLogin = geoVelocity
            lastLoginStatus = prevRows[i][LOGIN_TABLE_COLUMNS[1]]
        del prevRows[0]

        # add new feature row
        featureDf.at[j, FEATURE_COLUMNS[0]] = float(maxConsecutiveFailures) / float(WINDOW_SIZE)
        featureDf.at[j, FEATURE_COLUMNS[1]] = float(totalFailures) / float(WINDOW_SIZE)
        featureDf.at[j, FEATURE_COLUMNS[2]] = float(timeBetweenMaxCosecutiveFailures)
        featureDf.at[j, FEATURE_COLUMNS[3]] = float(timeBetweenLastTwoLogins)
        featureDf.at[j, FEATURE_COLUMNS[4]] = float(maxGeoVelocity)
        featureDf.at[j, FEATURE_COLUMNS[5]] = float(geoVelocityForLastLogin)
        featureDf.at[j, FEATURE_COLUMNS[6]] = float(lastLoginStatus)
        featureDf.at[j, FEATURE_COLUMNS[7]] = float(suspiciousLogin)
        j = j + 1


print featureDf

fileName = 'features-' +  datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.csv'
featureDf.to_csv(fileName)