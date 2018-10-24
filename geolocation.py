import yaml
import requests

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
        latitude = jsonObj[LATITUDE]
        longitude = jsonObj[LONGITUDE]
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