import yaml
import requests
import math

API_KEY = None
API_ENDPOINT = 'http://api.ipstack.com/'

LATITUDE = 'latitude'
LONGITUDE = 'longitude'
COUNTRY_NAME = 'country_name'
UNKNOWN = 'Unknown'

R_EARTH = 6378100.0

with open('config.yaml', 'r') as stream:
    try:
        config = yaml.load(stream)
        API_KEY = config['API_KEY']
    except yaml.YAMLError as exc:
        print(exc)


class GeoLocation:

    def __init__(self, latitude, longitude, country):
        self.latitude = latitude
        self.longitude = longitude
        self.country = country

    @classmethod
    def load_from_json(cls, jsonObj):
        if LATITUDE not in jsonObj or LONGITUDE not in jsonObj:
            raise ValueError('Location not found.')
        try:
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


def get_geolocation(ip):
    params = {
        'access_key': API_KEY
    }
    response = requests.get(API_ENDPOINT + ip, params=params)
    geolocation = GeoLocation.load_from_json(response.json())
    return geolocation


# calculate distance between two points given by longitude, latitude
def calculate_distance(longitude1, latitude1, longitude2, latitude2):
    longitude1 = math.radians(longitude1)
    latitude1 = math.radians(latitude1)
    longitude2 = math.radians(longitude2)
    latitude2 = math.radians(latitude2)
    dlon = longitude2 - longitude1
    dlat = latitude2 - latitude2
    a = math.sin(dlat / 2) ** 2 + math.cos(latitude1) * math.cos(latitude2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dist = R_EARTH * c
    if math.isnan(dist):
        return 0
    else:
        return dist


def distance(geolocation1, geolocation2):
    return calculate_distance(geolocation1.longitude, geolocation1.latitude, geolocation2.longitude,
                              geolocation2.latitude)
