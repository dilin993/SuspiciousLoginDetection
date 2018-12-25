import pandas as pd
import datetime
import random
from random import getrandbits
from ipaddress import IPv4Address
import constants
import uuid
import time
import numpy as np
import geolocation

NUMBER_OF_USERS = 100
ITERATIONS = 50000

usernames = None
currentTime = None
index = 0


def generate_random_ipv4():
    bits = getrandbits(32)  # generates an integer with 32 random bits
    addr = IPv4Address(bits)  # instances an IPv4Address object from those bits
    return str(addr)


ipAddressesForUsers = {}
geoLocationForIP = {}

index = 0


# Generate successful login
def generate_successful_login(df, given_time=None, username=None, context_id=None, ip=None, is_suspicious=False):
    global currentTime, usernames, ipAddressesForUsers, geoLocationForIP, index

    if username is None:
        username = usernames[random.randrange(NUMBER_OF_USERS)]
    event_time = given_time
    if event_time is None:
        currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3000))
        event_time = currentTime
    if context_id is None:
        context_id = uuid.uuid4()
    if ip is None:
        ip = ipAddressesForUsers[username]
    geoloc = geoLocationForIP[ip]

    df.at[index, constants.HEADER_USERNAME] = username
    df.at[index + 1, constants.HEADER_USERNAME] = username

    df.at[index, constants.HEADER_CONTEXTID] = context_id
    df.at[index + 1, constants.HEADER_CONTEXTID] = context_id

    df.at[index, constants.HEADER_EVENTTYPE] = constants.EVENTYPE_STEP
    df.at[index + 1, constants.HEADER_EVENTTYPE] = constants.EVENTYPE_OVERALL

    df.at[index, constants.HEADER_TIMESTAMP] = str(int(time.mktime(event_time.timetuple()) * 1e3))
    currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3))
    event_time = currentTime
    df.at[index + 1, constants.HEADER_TIMESTAMP] = str(int(time.mktime(event_time.timetuple()) * 1e3))

    df.at[index, constants.HEADER_REMOTEIP] = ip
    df.at[index + 1, constants.HEADER_REMOTEIP] = ip
    df.at[index, constants.HEADER_AUTHENTICATIONSUCCESS] = 0
    df.at[index + 1, constants.HEADER_AUTHENTICATIONSUCCESS] = 1

    if is_suspicious:
        df.at[index, constants.HEADER_SUSPICIOUS_LOGIN] = 1
        df.at[index + 1, constants.HEADER_SUSPICIOUS_LOGIN] = 1
    else:
        df.at[index, constants.HEADER_SUSPICIOUS_LOGIN] = 0
        df.at[index + 1, constants.HEADER_SUSPICIOUS_LOGIN] = 0

    df.at[index, constants.HEADER_LATITUDE] = geoloc.latitude
    df.at[index, constants.HEADER_LONGITUDE] = geoloc.longitude
    df.at[index, constants.HEADER_COUNTRY] = geoloc.country
    df.at[index + 1, constants.HEADER_LATITUDE] = geoloc.latitude
    df.at[index + 1, constants.HEADER_LONGITUDE] = geoloc.longitude
    df.at[index + 1, constants.HEADER_COUNTRY] = geoloc.country

    index = index + 2


# Generate failure login
def generate_failure_login(df, iterations=0, given_time=None, username=None, context_id=None, ip=None):
    global currentTime, usernames, ipAddressesForUsers, geoLocationForIP, index

    if username is None:
        username = usernames[random.randrange(NUMBER_OF_USERS)]
    event_time = given_time
    if event_time is None:
        currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3000))
        event_time = currentTime
    if context_id is None:
        context_id = uuid.uuid4()
    if ip is None:
        ip = ipAddressesForUsers[username]
    geoloc = geoLocationForIP[ip]

    for i in range(iterations + 1):
        df.at[index, constants.HEADER_USERNAME] = username

        df.at[index, constants.HEADER_CONTEXTID] = context_id

        df.at[index, constants.HEADER_EVENTTYPE] = constants.EVENTYPE_STEP

        df.at[index, constants.HEADER_TIMESTAMP] = str(int(time.mktime(event_time.timetuple()) * 1e3))
        currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3))
        event_time = currentTime

        df.at[index, constants.HEADER_REMOTEIP] = ipAddressesForUsers[username]

        df.at[index, constants.HEADER_AUTHENTICATIONSUCCESS] = 0
        df.at[index, constants.HEADER_SUSPICIOUS_LOGIN] = 0

        df.at[index, constants.HEADER_LATITUDE] = geoloc.latitude
        df.at[index, constants.HEADER_LONGITUDE] = geoloc.longitude
        df.at[index, constants.HEADER_COUNTRY] = geoloc.country

        index = index + 1


# Generate Suspicious login - Login after consecutive failed logins
def generate_suspicious_login_scenario1(df):
    global currentTime, usernames, ipAddressesForUsers, index
    username = usernames[random.randrange(NUMBER_OF_USERS)]
    context_id = uuid.uuid4()
    currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3000))
    failure_count = random.randrange(4, 7)
    generate_failure_login(df, failure_count, currentTime, username, context_id)
    index = index + failure_count
    currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3))
    generate_successful_login(df, currentTime, username, context_id, is_suspicious=True)
    index = index + 1


# Generate Suspicious login 2 - Login from suspicious IP
def generate_suspicious_login_scenario2(df):
    global currentTime, usernames, ipAddressesForUsers, index
    username = usernames[random.randrange(NUMBER_OF_USERS)]
    currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3000))
    ip = generate_random_ipv4()
    geo_loc = None
    while geo_loc is None:
        while ip in geoLocationForIP:
            ip = generate_random_ipv4()
        geo_loc = geolocation.get_geolocation(ip)
        if geo_loc.latitude == geoLocationForIP[ipAddressesForUsers[username]].latitude and geo_loc.longitude == \
                geoLocationForIP[ipAddressesForUsers[username]].longitude:
            geo_loc = None
            ip = generate_random_ipv4()
    geoLocationForIP[ip] = geo_loc
    generate_successful_login(df, currentTime, username)
    index = index + 1
    currentTime = currentTime + datetime.timedelta(seconds=random.randrange(60))
    generate_successful_login(df, currentTime, username, ip, is_suspicious=True)
    index = index + 1


LOGIN_TABLE_COLUMNS = [constants.HEADER_USERNAME, constants.HEADER_CONTEXTID, constants.HEADER_EVENTTYPE,
                       constants.HEADER_AUTHENTICATIONSUCCESS, constants.HEADER_REMOTEIP, constants.HEADER_TIMESTAMP,
                       constants.HEADER_LATITUDE,
                       constants.HEADER_LONGITUDE, constants.HEADER_COUNTRY, constants.HEADER_SUSPICIOUS_LOGIN]


def main():
    global currentTime, usernames, ipAddressesForUsers, geoLocationForIP, index, currentTime

    # generate usernames
    usernames = ['user' + str(x) for x in range(1, NUMBER_OF_USERS + 1)]

    # set current time
    currentTime = datetime.datetime.now()

    print("Generating IP addresses...")
    # generate an IP for each user
    for username in usernames:
        ipAddressesForUsers[username] = generate_random_ipv4()
        geoLocationForIP[ipAddressesForUsers[username]] = geolocation.get_geolocation(ipAddressesForUsers[username])

    login_data = pd.DataFrame(index=None, columns=LOGIN_TABLE_COLUMNS)
    index = 0
    iterations = ITERATIONS
    choices = [1, 2, 3, 4]
    p = [0.3, 0.3, 0.3, 0.1]
    print("Simulating user logins...")
    for i in range(iterations):
        c = np.random.choice(choices, p=p)
        if c == 1:
            generate_successful_login(login_data)
        elif c == 2:
            generate_failure_login(login_data)
        elif c == 3:
            generate_suspicious_login_scenario1(login_data)
        else:
            generate_suspicious_login_scenario2(login_data)
        print('Iteration ', i, ' of ', iterations, ' completed.')

    fileName = 'login_data-' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.csv'
    login_data.to_csv(fileName)
    print('Login data saved in \'' + fileName + '\'.')


if __name__ == '__main__':
    main()
