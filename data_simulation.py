import pandas as pd
import datetime
import random
from random import getrandbits
from ipaddress import IPv4Address, IPv6Address
import constants
import uuid
import time
import numpy as np

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

ipAddressesforUsers = {}

# generate an IP for each user
for username in usernames:
    ipAddressesforUsers[username] = generateRandomIPV4()


### Generate successful login
def generateSuccessfulLogin(df, givenTime=None, username=None, contextID=None):
    global currentTime, usernames, ipAddressesforUsers, index

    if username is None:
        username = usernames[random.randrange(NUMBER_OF_USERS)]
    eventTime = givenTime
    if eventTime is None:
        currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3000))
        eventTime = currentTime
    if contextID is None:
        contextID = uuid.uuid4()
    
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

    df.at[index, constants.HEADER_REMOTEIP] = ipAddressesforUsers[username]
    df.at[index+1, constants.HEADER_REMOTEIP] = ipAddressesforUsers[username]

    df.at[index, constants.HEADER_AUTHENTICATIONSUCCESS] = 0
    df.at[index+1, constants.HEADER_AUTHENTICATIONSUCCESS] = 1

    index = index + 2

### Generate failed login
def generateFailedLogin(df, iter=0, givenTime=None, username=None, contextID=None):
    global currentTime, usernames, ipAddressesforUsers, index

    if username is None:
        username = usernames[random.randrange(NUMBER_OF_USERS)]
    eventTime = givenTime
    if eventTime is None:
        currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3000))
        eventTime = currentTime
    if contextID is None:
        contextID = uuid.uuid4()

    for i in range(iter+1):
        df.at[index, constants.HEADER_USERNAME] = username

        df.at[index, constants.HEADER_CONTEXTID] = contextID

        df.at[index, constants.HEADER_EVENTTYPE] = constants.EVENTYPE_STEP

        df.at[index, constants.HEADER_TIMESTAMP] = str(int(time.mktime(eventTime.timetuple())*1e3))
        currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3))
        eventTime = currentTime

        df.at[index, constants.HEADER_REMOTEIP] = ipAddressesforUsers[username]

        df.at[index, constants.HEADER_AUTHENTICATIONSUCCESS] = 0

        index = index + 1


### Generate failed login
def generateSuspiciousLoginScenario1(df, givenTime=None):
    global currentTime, usernames, ipAddressesforUsers, index
    username = usernames[random.randrange(NUMBER_OF_USERS)]
    contextID = uuid.uuid4()
    currentTime = currentTime + datetime.timedelta(seconds=random.randrange(3000))
    failureCount = random.randrange(4,7)
    generateFailedLogin(df, failureCount, currentTime, username, contextID)
    index = index + failureCount
    generateSuccessfulLogin(df, givenTime, username, contextID)

LOGIN_TABLE_COLUMNS = [constants.HEADER_USERNAME, constants.HEADER_CONTEXTID, constants.HEADER_EVENTTYPE,
 constants.HEADER_AUTHENTICATIONSUCCESS, constants.HEADER_REMOTEIP, constants.HEADER_TIMESTAMP]

loginData = pd.DataFrame(index=None, columns=LOGIN_TABLE_COLUMNS)
index = 0
iterations = 10000
choices = [1,2,3]
p = [0.6, 0.39, 0.01]
for i in range(iterations):
    c = np.random.choice(choices, p=p)
    if c == 1:
        generateSuccessfulLogin(loginData)
    elif c == 2:
        generateFailedLogin(loginData)
    else:
        generateSuspiciousLoginScenario1(loginData)

print loginData

loginData.to_csv('loginData.csv')

    
    
    
