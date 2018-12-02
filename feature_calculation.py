import queue
import numpy as np
import datetime
import geolocation

USER_USERNAME = 'username'
USER_CONTEXTID = 'contextId'
USER_EVENTTYPE = 'eventType'
USER_TIMESTAMP = '_timestamp'
USER_REMOTEIP = 'remoteIp'
USER_AUTHENTICATIONSUCCESS = 'authenticationSuccess'
USER_LATITUDE = 'latitude'
USER_LONGITUDE = 'longitude'
USER_COUNTRY = 'country'
USER_SUSPICIOUS_LOGIN = 'suspiciousLogin'

EVENTYPE_STEP = 'step'
EVENTYPE_OVERALL = 'overall'

WINDOW_SIZE = 5

geoLocationForIP = {}
loginData = {}
prevLogin = {}

MAX_TIME = 300
MAX_GEOVELOCITY = 400


class UserLogin:
    loginSuccess = False
    timestamp = None
    ip = ''
    contextID = ''
    isSuspicious = False

    def __init__(self, username):
        self.username = username

    def __str__(self):
        tmpStr = ""
        tmpStr = tmpStr + "\nUser: " + self.username + "\n"
        tmpStr = tmpStr + "loginSucess: " + str(self.loginSuccess) + "\n\n"
        return tmpStr

    def getUsername(self):
        return self.username

    def setGeolocation(self, ip):
        self.ip = ip
        if ip not in geoLocationForIP:
            geoLocationForIP[ip] = geolocation.getGelocation(ip)

    def getGeolocation(self):
        if self.ip == '':
            raise Exception("IP not set!")
        if self.ip not in geoLocationForIP:
            raise Exception("IP entry not found!")
        return geoLocationForIP[self.ip]


# Insert given payload data to loginData
def insertToLoginData(data):

    if USER_USERNAME not in data:
        raise Exception("Invalid payload data!")
    username = data[USER_USERNAME]

    # not enough data
    if username not in prevLogin:
        prevLogin[username] = data
        return
    prev = prevLogin[username]

    userLogin = UserLogin(username)

    if prev[USER_CONTEXTID] == data[USER_CONTEXTID] and prev[USER_EVENTTYPE] == EVENTYPE_STEP and \
            data[USER_EVENTTYPE] == EVENTYPE_OVERALL and data[USER_AUTHENTICATIONSUCCESS]:
        userLogin.loginSuccess = True
        userLogin.timestamp = datetime.datetime.fromtimestamp(data[USER_TIMESTAMP] / 1e3)
        userLogin.setGeolocation(data[USER_REMOTEIP])
        userLogin.contextID = data[USER_CONTEXTID]
    elif prev[USER_EVENTTYPE] == EVENTYPE_STEP and not prev[USER_AUTHENTICATIONSUCCESS]:
        userLogin.loginSuccess = False
        userLogin.timestamp = datetime.datetime.fromtimestamp(prev[USER_TIMESTAMP] / 1e3)
        userLogin.setGeolocation(prev[USER_REMOTEIP])
        userLogin.contextID = prev[USER_CONTEXTID]
    else:
        prevLogin[username] = data
        return

    if username not in loginData:
        loginData[username] = queue.Queue(maxsize=WINDOW_SIZE)
    else:
        if loginData[username].full():
            loginData[username].get()
    loginData[username].put(userLogin)
    print(userLogin)
    prevLogin[username] = data


def getTimeDiff(timestamp1, timestamp2):
    diff = abs((timestamp1 - timestamp2).seconds)
    return diff

# Calculate the features of current user and returns a numpy array
def getFeatures(username):

    if username not in loginData:
        return None
    userData = list(loginData[username].queue)
    if len(userData) < WINDOW_SIZE:
        return None
    feature = []
    prevConsecutiveFailures = 0
    consecutiveFailureTime = 0.0
    loginSucess = 0
    lastGeoVelocity = 0.0
    maxGeoVelocity = 0
    numFailures = 0
    ipChangedLastTime = 0

    if userData[WINDOW_SIZE - 1].loginSuccess:
        loginSucess = 1
    curContextID = userData[WINDOW_SIZE - 1].contextID
    curTime = userData[WINDOW_SIZE - 1].timestamp
    lastTimeDiff = getTimeDiff(curTime, userData[WINDOW_SIZE - 2].timestamp)
    lastDist = abs(geolocation.distance(userData[WINDOW_SIZE - 1].getGeolocation(),
                                        userData[WINDOW_SIZE - 2].getGeolocation()))

    if lastDist != 0:
        lastGeoVelocity = lastDist / lastTimeDiff

    if userData[WINDOW_SIZE - 1].ip != userData[WINDOW_SIZE - 2].ip:
        ipChangedLastTime = 1

    for i in range(WINDOW_SIZE - 2, -1, -1):
        if userData[i].contextID == curContextID:
            prevConsecutiveFailures = prevConsecutiveFailures + 1
            consecutiveFailureTime = getTimeDiff(curTime, userData[i].timestamp)
        else:
            break

    for i in range(WINDOW_SIZE):
        if i > 0:
            dt = getTimeDiff(userData[i - 1].timestamp, userData[i].timestamp)
            dist = abs(geolocation.distance(userData[i-1].getGeolocation(), userData[i].getGeolocation()))
            geoVelocity = 0
            if dist != 0 and dt != 0:
                geoVelocity = dist / dt
            if maxGeoVelocity < geoVelocity:
                maxGeoVelocity = geoVelocity

        if not userData[i].loginSuccess:
            numFailures = numFailures + 1

    # add new feature row
    feature.append(float(min(lastGeoVelocity, MAX_GEOVELOCITY)) / float(MAX_GEOVELOCITY))
    feature.append(float(prevConsecutiveFailures) / float(WINDOW_SIZE))
    feature.append(float(loginSucess))
    feature.append(float(min(consecutiveFailureTime, MAX_TIME)) / float(MAX_TIME))
    # feature.append(float(min(maxGeoVelocity, MAX_GEOVELOCITY)) / float(MAX_GEOVELOCITY))
    feature.append(float(ipChangedLastTime))
    feature.append(float(numFailures) / float(WINDOW_SIZE))

    return np.array(feature, dtype=np.float64)

