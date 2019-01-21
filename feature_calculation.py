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
STEP_AUTHENTICATOR = 'stepAuthenticator'
AUTHENTICATION_STEP = 'authenticationStep'

BASIC_AUTHENTICATOR = 'BasicAuthenticator'

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
        tmpstr = ""
        tmpstr = tmpstr + "\nUser: " + self.username + "\n"
        tmpstr = tmpstr + "loginSucess: " + str(self.loginSuccess) + "\n\n"
        return tmpstr

    def get_username(self):
        return self.username

    def set_geolocation(self, ip):
        self.ip = ip
        if ip not in geoLocationForIP:
            geoLocationForIP[ip] = geolocation.get_geolocation(ip)

    def get_geolocation(self):
        if self.ip == '':
            raise Exception("IP not set!")
        if self.ip not in geoLocationForIP:
            raise Exception("IP entry not found!")
        return geoLocationForIP[self.ip]

    def set_geolocation_from_data(self, data):
        try:
            self.ip = data[USER_REMOTEIP]
            latitude = data[USER_LATITUDE]
            longitude = data[USER_LONGITUDE]
            country = data[USER_COUNTRY]
            if self.ip not in geoLocationForIP:
                geoLocationForIP[self.ip] = geolocation.GeoLocation(latitude, longitude, country)
        except KeyError:
            if self.ip != '':
                self.set_geolocation(self.ip)

    def flag_suspicious(self, data):
        if USER_SUSPICIOUS_LOGIN in data:
            suspicious_login = str(data[USER_SUSPICIOUS_LOGIN]).lower()
            if suspicious_login == '1':
                self.isSuspicious = True
            else:
                self.isSuspicious = False


# Insert given payload data to loginData
def insert_to_login_data(data):

    if USER_USERNAME not in data:
        raise Exception("Invalid payload data!")
    if AUTHENTICATION_STEP in data and STEP_AUTHENTICATOR in data:
        if data[AUTHENTICATION_STEP] == '1' and data[STEP_AUTHENTICATOR] != BASIC_AUTHENTICATOR:
            return  # skip anything other than basic authenticator

    username = data[USER_USERNAME]

    # not enough data
    if username not in prevLogin:
        prevLogin[username] = data
        return
    prev = prevLogin[username]

    userlogin = UserLogin(username)

    if prev[USER_CONTEXTID] == data[USER_CONTEXTID] and prev[USER_EVENTTYPE] == EVENTYPE_STEP and \
            data[USER_EVENTTYPE] == EVENTYPE_OVERALL and data[USER_AUTHENTICATIONSUCCESS]:
        userlogin.loginSuccess = True
        userlogin.timestamp = datetime.datetime.fromtimestamp(data[USER_TIMESTAMP] / 1e3)
        userlogin.set_geolocation_from_data(data)
        userlogin.contextID = data[USER_CONTEXTID]
        userlogin.flag_suspicious(data)
    elif prev[USER_EVENTTYPE] == EVENTYPE_STEP and not prev[USER_AUTHENTICATIONSUCCESS]:
        userlogin.loginSuccess = False
        userlogin.timestamp = datetime.datetime.fromtimestamp(prev[USER_TIMESTAMP] / 1e3)
        userlogin.set_geolocation_from_data(prev)
        userlogin.contextID = prev[USER_CONTEXTID]
        userlogin.flag_suspicious(prev)
    else:
        prevLogin[username] = data
        return

    if username not in loginData:
        loginData[username] = queue.Queue(maxsize=WINDOW_SIZE)
    else:
        if loginData[username].full():
            loginData[username].get()
    loginData[username].put(userlogin)
    print(userlogin)
    prevLogin[username] = data


def get_time_diff(timestamp1, timestamp2):
    diff = abs((timestamp1 - timestamp2).seconds)
    return diff


# Calculate the features of current user and returns a numpy array
def get_features(username, append_label=False):

    if username not in loginData:
        return None
    userdata = list(loginData[username].queue)
    if len(userdata) < WINDOW_SIZE:
        return None
    feature = []
    prev_consecutive_failures = 0
    consecutive_failure_time = 0.0
    login_success = 0
    last_geo_velocity = 0.0
    max_geo_velocity = 0
    num_failures = 0
    ip_changed_last_time = 0

    if userdata[WINDOW_SIZE - 1].loginSuccess:
        login_success = 1
    cur_context_id = userdata[WINDOW_SIZE - 1].contextID
    cur_time = userdata[WINDOW_SIZE - 1].timestamp
    last_time_diff = get_time_diff(cur_time, userdata[WINDOW_SIZE - 2].timestamp)
    last_dist = abs(geolocation.distance(userdata[WINDOW_SIZE - 1].get_geolocation(),
                                        userdata[WINDOW_SIZE - 2].get_geolocation()))

    if last_dist != 0:
        last_geo_velocity = last_dist / last_time_diff

    if userdata[WINDOW_SIZE - 1].ip != userdata[WINDOW_SIZE - 2].ip:
        ip_changed_last_time = 1

    for i in range(WINDOW_SIZE - 2, -1, -1):
        if userdata[i].contextID == cur_context_id:
            prev_consecutive_failures = prev_consecutive_failures + 1
            consecutive_failure_time = get_time_diff(cur_time, userdata[i].timestamp)
        else:
            break

    for i in range(WINDOW_SIZE):
        if i > 0:
            dt = get_time_diff(userdata[i - 1].timestamp, userdata[i].timestamp)
            dist = abs(geolocation.distance(userdata[i-1].get_geolocation(), userdata[i].get_geolocation()))
            geo_velocity = 0
            if dist != 0 and dt != 0:
                geo_velocity = dist / dt
            if max_geo_velocity < geo_velocity:
                max_geo_velocity = geo_velocity

        if not userdata[i].loginSuccess:
            num_failures = num_failures + 1

    # add new feature row
    feature.append(float(min(last_geo_velocity, MAX_GEOVELOCITY)) / float(MAX_GEOVELOCITY))
    feature.append(float(prev_consecutive_failures) / float(WINDOW_SIZE))
    feature.append(float(login_success))
    feature.append(float(min(consecutive_failure_time, MAX_TIME)) / float(MAX_TIME))
    # feature.append(float(min(maxGeoVelocity, MAX_GEOVELOCITY)) / float(MAX_GEOVELOCITY))
    feature.append(float(ip_changed_last_time))
    feature.append(float(num_failures) / float(WINDOW_SIZE))
    if append_label:
        if userdata[WINDOW_SIZE-1].isSuspicious:
            feature.append(1)
        else:
            feature.append(0)

    return np.array(feature, dtype=np.float64)

