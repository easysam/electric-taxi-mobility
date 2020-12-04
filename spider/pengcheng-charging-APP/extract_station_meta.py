#coding=gbk
import json
import requests
import os
import os.path
import pandas as pd

citys=["shenzhen"]
for city in citys:
    dir = r"./station_meta"
    if not os.path.exists(dir):
        os.mkdir(dir)
    file = open(os.path.join(dir, city + '_station_meta.csv'), 'w+')
    file.write("name,longitude,latitude,address,isConnected,dcNum,acNum"+"\n")
    stations = []
    db_path = './chargeApp'
    data_list = sorted(os.listdir(db_path))
    for i in data_list:
        with open(db_path + '/' + str(i), 'r', encoding='UTF-8') as obj:
            dict = json.load(obj)
        all = dict['data']['records']
        if all!=[]:
            id=""
            name = ""
            lon = ""
            lat = ""
            dcTotalNum = -1
            gcjlat = ""
            gcjlon = ""
            address=""
            # find the meta data (name id lon lat address)
            for a in all:
                for k, v in a.items():
                    if (k == "name"):
                        name = v
                    elif(k=="id"):
                        id = v
                    elif (k == "longitude"):
                        lon = v
                    elif (k == "latitude"):
                        lat = v
                    elif (k == "address"):
                        address = v
                # start spider
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/61.0.3163.100 Safari/537.36"}

                content = requests.get("https://app.sz-charge.cn/csmlt/serviceapi/station/queryStationDetail?id=" + str(
                    id) + "&latitude=" + str(lat) + "&longitude=" + str(lon),
                                       headers=headers, verify=False, allow_redirects=False)
                response = content.json()
                # find whether the station is connected to the Internet
                dc = 0
                ac = 0
                connected = False
                if (response['data']['totalConnectorNum'] != None):
                    connected = True
                    charge = response['data']['chargeTypes']
                    for a, b in charge[0].items():
                        if (a == "total") & (b != None):
                            dc = b
                    for a, b in charge[1].items():
                        if (a == "total") & (b != None):
                            ac = b
                else:
                    content2 = requests.get("https://app.sz-charge.cn/csmlt/serviceapi/station/queryConnectorDetails?stationId="+str(id)+"&pageStart=1&pageSize=20"
                        ,headers=headers, verify=False, allow_redirects=False)
                    response2 = content2.json()
                    pagenum=0
                    # total number of the charging piles which not connected to Internet
                    total=int(response2['data']['totalRecord'])
                    if total%20 == 0:
                        pagenum=total//20
                    else:
                        pagenum=(total//20)+1
                    for page in range(1,pagenum+1):

                        content3=requests.get("https://app.sz-charge.cn/csmlt/serviceapi/station/queryConnectorDetails?stationId="+str(id)+"&pageStart="+str(page)+"&pageSize=20"
                        ,headers=headers, verify=False, allow_redirects=False)
                        response3 = content3.json()
                        records=response3['data']['records']
                        for record in records:
                            for c,d in record.items():
                                if (c=="type")&(d==4):
                                    dc+=1
                                elif(c=="type")&(d==3):
                                    ac+=1
                file.write(name + "," + str(lon) + "," + str(lat) + "," + str(address) + "," + str(connected) + "," + str(
                    dc)+ ","+str(ac) + "\n")
    file.close()
    path=os.path.join(dir, city + '_station_meta.csv')
    df=pd.read_csv(path,encoding='gbk')
    df= df.drop_duplicates( keep='first')
    df.to_csv(path,index=False)





