#coding=gbk
import json
import os
import os.path
import pandas as pd

citys=["shenzhen"]
for city in citys:
    dir = r"./station_meta"
    if not os.path.exists(dir):
        os.mkdir(dir)
    file = open(os.path.join(dir, city + '_station_meta.csv'), 'w+')
    file.write("name,longitude,latitude,gcj02Longitude,gcj02Latitude,dcTotalNum"+"\n")
    stations = []
    db_path = './chargeApp'
    data_list = sorted(os.listdir(db_path))
    for i in data_list:
        with open(db_path + '/' + str(i), 'r', encoding='UTF-8') as obj:
            dict = json.load(obj)
        all = dict['result']
        if all != []:
            name = ""
            lon = ""
            lat = ""
            dcTotalNum = -1
            gcjlat = ""
            gcjlon = ""
            for a in all:
                print(a)
                for k, v in a.items():
                    if (k == "name"):
                        name = v
                    elif (k == "longitude"):
                        lon = v
                    elif (k == "latitude"):
                        lat = v
                    elif (k == "dcTotalNum"):
                        dcTotalNum = v
                    elif (k == "gcj02Latitude"):
                        gcjlat = v
                    elif (k == "gcj02Longitude"):
                        gcjlon = v
                file.write(name+","+str(lon)+","+str(lat)+","+str(gcjlon)+","+str(gcjlat)+","+str(dcTotalNum)+"\n")
    file.close()
    path=os.path.join(dir, city + '_station_meta.csv')
    df=pd.read_csv(path,encoding='gbk')
    df= df.drop_duplicates( keep='first')
    df.to_csv(path,index=False)





