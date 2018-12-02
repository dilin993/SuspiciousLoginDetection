import yaml
import requests
import math
import numpy as np

API_KEY = None
API_ENDPOINT = 'http://api.ipstack.com/'

LATITUDE = 'latitude'
LONGITUDE = 'longitude'
COUNTRY_NAME = 'country_name'
UNKNOWN = 'Unknown'

with open('config.yaml', 'r') as stream:
    try:
        config =  yaml.load(stream)
        API_KEY = config['API_KEY']
    except yaml.YAMLError as exc:
        print(exc)

class GeoLocation:

    def __init__(self, latitude, longitude, country):
        self.latitude = latitude
        self.longitude = longitude
        self.country = country

    @classmethod
    def loadFromJson(cls, jsonObj):
        if LATITUDE not in jsonObj or LONGITUDE not in jsonObj:
            raise ValueError('Location not found.')
        try :
            latitude = float(jsonObj[LATITUDE])
            longitude = float(jsonObj[LONGITUDE])
        except TypeError as e:
            latitude = 0
            longitude = 0
        if COUNTRY_NAME in jsonObj:
            country = jsonObj[COUNTRY_NAME]
        else:
            country = 'Unknown'
        return cls(latitude, longitude, country)


def getGelocation(ip):
    params = {
        'access_key': API_KEY
    }
    response = requests.get(API_ENDPOINT + ip, params=params)
    geolocation =  GeoLocation.loadFromJson(response.json())
    return geolocation


# calculate distance between two points given by longitude, latitude
def calculateDistance(longitudeA, latitudeA, longitudeB, latitudeB):
    longitudeA = math.radians(longitudeA)
    latitudeA = math.radians(latitudeA)
    longitudeB = math.radians(longitudeB)
    latitudeB = math.radians(latitudeB)
    R_EARTH = 6378100.0
    dlon = longitudeB - longitudeA
    dlat = latitudeB - latitudeB
    a = math.sin(dlat / 2) ** 2 + math.cos(latitudeA) * math.cos(latitudeB) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dist = R_EARTH * c
    if math.isnan(dist):
        return 0
    else:
        return dist


def distance(geolocationA, geolocationB):
    return calculateDistance(geolocationA.longitude, geolocationA.latitude, geolocationB.longitude, geolocationB.latitude)
