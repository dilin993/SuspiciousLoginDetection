import pandas as pd
import datetime
import random
from random import getrandbits
from ipaddress import IPv4Address, IPv6Address
import constants
import uuid
import time
import numpy as np
import geolocation

NUMBER_OF_USERS = 100

# generate usernames
usernames = ['user' + str(x) for x in range(1, NUMBER_OF_USERS + 1)]

# set current time
currentTime = datetime.datetime.now()


def generateRandomIPV4():
    bits = getrandbits(32) # generates an integer with 32 random bits
    addr = IPv4Address(bits) # instances an IPv4Address object from those bits
    addrStr = str(addr) # get the IPv4Address object's string representation
    return addrStr

ipAddressesForUsers = {}
geolocationForIP = {}

index = 0

# generate an IP for each user
for username in usernames:
    ipAddressesForUsers[username] = generateRandomIPV4()
    geolocationForIP[ipAddressesForUsers[username]] = geolocation.getGelocation(ipAddressesForUsers[username])


### Generate successful login
def generateSuccessfulLogin(df, givenTime=None, username=None, contextID=None, ip=None, isSuspicious=False):
    global currentTime, usernames, ipAddressesForUsers, geolocationForUsers, index

    if username is None:
        username = usernames[random.randrange(NUMBER_OF_USERS)]
    eventTime = givenTime
    if eventTime is None:
        currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3000))
        eventTime = currentTime
    if contextID is None:
        contextID = uuid.uuid4()
    if ip is None:
        ip = ipAddressesForUsers[username]
    geoloc = geolocationForIP[ip]
    
    contextID = uuid.uuid4()
    
    df.at[index, constants.HEADER_USERNAME] = username
    df.at[index+1, constants.HEADER_USERNAME] = username

    df.at[index, constants.HEADER_CONTEXTID] = contextID
    df.at[index+1, constants.HEADER_CONTEXTID] = contextID

    df.at[index, constants.HEADER_EVENTTYPE] = constants.EVENTYPE_STEP
    df.at[index+1, constants.HEADER_EVENTTYPE] = constants.EVENTYPE_OVERALL

    df.at[index, constants.HEADER_TIMESTAMP] = str(int(time.mktime(eventTime.timetuple())*1e3))
    currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3))
    eventTime = currentTime
    df.at[index+1, constants.HEADER_TIMESTAMP] = str(int(time.mktime(eventTime.timetuple())*1e3))

    df.at[index, constants.HEADER_REMOTEIP] = ip
    df.at[index+1, constants.HEADER_REMOTEIP] = ip
    df.at[index, constants.HEADER_AUTHENTICATIONSUCCESS] = 0
    df.at[index+1, constants.HEADER_AUTHENTICATIONSUCCESS] = 1

    if isSuspicious:
        df.at[index, constants.HEADER_SUSPICIOUS_LOGIN] = 1
        df.at[index + 1, constants.HEADER_SUSPICIOUS_LOGIN] = 1
    else:
        df.at[index, constants.HEADER_SUSPICIOUS_LOGIN] = 0
        df.at[index + 1, constants.HEADER_SUSPICIOUS_LOGIN] = 0

    df.at[index, constants.HEADER_LATITUDE] = geoloc.latitude
    df.at[index, constants.HEADER_LONGITUDE] = geoloc.longitude
    df.at[index, constants.HEADER_COUNTRY] = geoloc.country
    df.at[index+1, constants.HEADER_LATITUDE] = geoloc.latitude
    df.at[index+1, constants.HEADER_LONGITUDE] = geoloc.longitude
    df.at[index+1, constants.HEADER_COUNTRY] = geoloc.country

    index = index + 2

### Generate failed login
def generateFailedLogin(df, iter=0, givenTime=None, username=None, contextID=None, ip=None):
    global currentTime, usernames, ipAddressesForUsers, geolocationForUsers, index

    if username is None:
        username = usernames[random.randrange(NUMBER_OF_USERS)]
    eventTime = givenTime
    if eventTime is None:
        currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3000))
        eventTime = currentTime
    if contextID is None:
        contextID = uuid.uuid4()
    if ip is None:
        ip = ipAddressesForUsers[username]
    geoloc = geolocationForIP[ip]

    for i in range(iter+1):
        df.at[index, constants.HEADER_USERNAME] = username

        df.at[index, constants.HEADER_CONTEXTID] = contextID

        df.at[index, constants.HEADER_EVENTTYPE] = constants.EVENTYPE_STEP

        df.at[index, constants.HEADER_TIMESTAMP] = str(int(time.mktime(eventTime.timetuple())*1e3))
        currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3))
        eventTime = currentTime

        df.at[index, constants.HEADER_REMOTEIP] = ipAddressesForUsers[username]

        df.at[index, constants.HEADER_AUTHENTICATIONSUCCESS] = 0
        df.at[index, constants.HEADER_SUSPICIOUS_LOGIN] = 0

        df.at[index, constants.HEADER_LATITUDE] = geoloc.latitude
        df.at[index, constants.HEADER_LONGITUDE] = geoloc.longitude
        df.at[index, constants.HEADER_COUNTRY] = geoloc.country

        index = index + 1


### Generate Suspicious login - Login after consecutive failed logins
def generateSuspiciousLoginScenario1(df, givenTime=None):
    global currentTime, usernames, ipAddressesForUsers, index
    username = usernames[random.randrange(NUMBER_OF_USERS)]
    contextID = uuid.uuid4()
    currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3000))
    failureCount = random.randrange(4,7)
    generateFailedLogin(df, failureCount, currentTime, username, contextID)
    index = index + failureCount
    generateSuccessfulLogin(df, givenTime, username, contextID,isSuspicious=True)
    index = index + 1


### Generate Suspicious login 2 - Login from suspicious IP
def generateSuspiciousLoginScenario2(df, givenTime=None):
    global currentTime, usernames, ipAddressesForUsers, index
    username = usernames[random.randrange(NUMBER_OF_USERS)]
    contextID = uuid.uuid4()
    currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3000))
    ip = generateRandomIPV4()
    while ip in geolocationForIP:
        ip = generateRandomIPV4()
    geolocationForIP[ip] = geolocation.getGelocation(ip)
    generateSuccessfulLogin(df, givenTime, username, contextID)
    index = index + 1
    generateSuccessfulLogin(df, givenTime, username, contextID, ip, isSuspicious=True)
    index = index + 1

LOGIN_TABLE_COLUMNS = [constants.HEADER_USERNAME, constants.HEADER_CONTEXTID, constants.HEADER_EVENTTYPE,
 constants.HEADER_AUTHENTICATIONSUCCESS, constants.HEADER_REMOTEIP, constants.HEADER_TIMESTAMP, constants.HEADER_LATITUDE,
 constants.HEADER_LONGITUDE, constants.HEADER_COUNTRY, constants.HEADER_SUSPICIOUS_LOGIN]

loginData = pd.DataFrame(index=None, columns=LOGIN_TABLE_COLUMNS)
index = 0
iterations = 5000
choices = [1, 2, 3, 4]
p = [0.6, 0.38, 0.01, 0.01]
for i in range(iterations):
    c = np.random.choice(choices, p=p)
    if c == 1:
        generateSuccessfulLogin(loginData)
    elif c == 2:
        generateFailedLogin(loginData)
    elif c == 3:
        generateSuspiciousLoginScenario1(loginData)
    else:
        generateSuspiciousLoginScenario2(loginData)

print loginData

fileName = 'loginData-' +  datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.csv'
loginData.to_csv(fileName)
