import json
import urllib.request

if __name__ == '__main__':
    contents = urllib.request.urlopen(
        "http://router.project-osrm.org/route/v1/driving/116.35155,39.95679;116.34748,39.93943?overview=false").read()
    my_json = contents.decode('utf8').replace("'", '"')
    dis = json.loads(my_json)['routes'][0]['distance']
    duration = json.loads(my_json)['routes'][0]['duration']
