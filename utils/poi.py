import requests
import csv
import os
import argparse
import time
import numpy as np
import pandas as pd

# url
url1 = "https://restapi.amap.com/v3/place/around?types="
url2 = "&location="
url3 = "&distance=1000&output=JSON&offset=25&extensions=all&page="
url4 = "&key="  # todo

types = {
    'poi':
        ["090100",  # 医疗保健-医院
         "110000",  # 风景名胜
         "130100",  # 政府机构及社会团体-政府机构
         "150100",  # 交通设施-机场
         "150500",  # 交通设施-地铁站
         "150700",  # 交通设施-公交车站
         "160100",  # 金融保险-银行
         "170100",  # 公司企业-知名企业
         "141201",  # 科教文化-高等院校
         "120300",  # 商务住宅-住宅区
         "100100",  # 住宿服务-宾馆酒店
         "060400",  # 购物服务-超市
         "050300"  # 餐饮服务-快餐
         ],
    'commerce':
        [
         "060400",  # 购物服务-超级市场
         "060100"   # 购物服务-商场
         ],
    'transportation':
        ["150500",  # 地铁
         "150700",  # 公交
         "150903",  # 换乘停车场
         "150904",  # 公共停车场
         "150905",  # 公用停车场
         "150906",  # 路边停车场
         ]
}

key_set = ["ae137009218a988d9e72c6a47c940a5a",
           "9331f8187dff43fff6822be528c919c5",
           "073ea5f185a8ddb1dde740c62723c580",
           "da6061f9d8057a6609ed838b461ec1bb",
           "b4baad21b9015c33121bd8766bfdd52f",
           "6e57a518f6d3c073ea37d5be185a1084",
           "678334132006841586bdf3f395af3a25",
           "d29f2f650a701e405d77f25c5ce42e3a",
           "fd2b818bb0e689f741f7fd5ffefe1500",
           "0219079df40faee23aafeeaed173c3fa",
           ##################################
           "60dfc3d04c70fe416d878c18c3e407aa",
           "590a26ce0248fe040ba91e743cfc176b",
           "a99d9dee94d6bee7345578dea71efe3d",
           "ec872a9dd0a83d5d9d631b52bbdc64c3",
           "3c6c89b46a2fdc6eda39e32e2638ecfa",
           "82bbdd7e3eefde3d4ba4a0286e4a0e18",
           "6e935107b265b687bee015df1baf6572",
           "82379610454706f5dc4ec6dbd7579bca",
           "d09b5d70fdf9119b973d5f31ac5122cf",
           "58a29558151fc430b3a1849e87560af2",
           ##################################
           "a4e8b71a22805479ea934738c7fd105e",
           "38e130f0c2033c35620314ded6278517",
           "630b7686e84e5e1bdd4dfdac801c5208",
           "f940043278e10ac8a891b9b491af670f",
           "72350f0c22f6ea308a70bf1ad250a877",
           "b1a2a97d172f587d1f1ab592957ec7a4",
           "9daebe038cf8b88fff0d0e02938f4857",
           "3b2006bd0e96ad6c77600ee40ba1ef95",
           "3f56477b09e0c3215e86d89df6612a43",
           "f1565c483d4bc974c0b88164bba1b704",
           ##################################
           "3a9323ad191e98605611b4589b87d12f",
           "fd6311f58bde31ae67676bae2d32d9df",
           "9395271a4fd90cc4e4ea7dce6ae34849",
           "aed5c3e736f0930fb0ed63858d247eb9",
           "e2c63175fbb24ae451f7bb40cde854fb",
           "c3463b3acac03d28aa3f5118d218aaa3",
           "8c94a8809c52cf47f41c259530335240",
           "ba3abb9a686bef42753dc4d230542bfb",
           "7069a533cb3526cb75497eed2a064db5",
           "121b162a69f4bb43d0d4671a53a97f95",
           ##################################
           "53328fc77cc618b041c3d8d0a4bfb271",
           "1ad11aa72eff5cd5b417dec479db5ac2",
           "a5bc0d0383d7c8fe925259d243e25bca",
           "2eb6f5289f572724fcc03bb9355b59fe",
           "e8c1d7b931547e8c463bb842337d2566",
           "2e32b227deb5aad8b9b2ae1fafac7d99",
           "832f9085f97717aa481a0876a407ee3c",
           "ec154d56fd214cbdc9a288f81b57def7",
           "b8f5a7edeebc6d00857983af0657be3a",
           "5c05161a30bd5e0d1be894e2e6323722",
           ##################################
           "9d79c4b674d1d6d2520f5b7d17c51b91",
           "4b2ea0f2f115e740bd0075e5fab1f3ca",
           "833341fc6cd363493fab5da4049a5928",
           "ff86c607b3287e567ef78fef710a292d",
           "04869e8900e09f5293f033b24a533286",
           "9146d1a226864ec9a2f76ef9ff8c2af1",
           "f542e92e4c8b37e29bcdf72d9d77e2a0",
           "9f6bd651710c59fc2068d76873a2d5cc",
           "dbf17b689eab23c57594ce8535ebded6",
           "90fb5915f774564cfea0860aaa760011",
           ##################################
           "4bd801b647ee723c6ba23b49ce8469bd",
           "df9f357a73a79401bf0c63fe9cdb798a",
           "5c0ebbf16699de6ad5233dc60cd37793",
           "ae8683704bb3eac0b7df71225b296940",
           "41801dfb71b85389ecf8b3d93e58dc4b",
           "a4014285eaad4e51aafe0f1f0e9feabb",
           "a5a9de0a53d5e99805a129a04ffa1bb4",
           "168dcdb12659eba59deec9262810c7b6",
           "857d5b3010164c4ea8b5cee892e3738d",
           "c7a8f6332727ddd8412a83693481a01a",
           ##################################
           "3a6b3dabf97e86e45059580384a6bcaa",
           "1eceab6f3f016d942986f4bf7a07d675",
           "95e3df1e44bdcbbc7c5e9f4a1f39107b",
           "ec03260d64a749eea31f1fd5dd5b2e74",
           "8671fd9b2e43a64d43d7a63ba545017d",
           "d9f92fd9e398da5765d717276f18d887",
           "949913f4d5da73e43d45b3af33a39834",
           "2f2d5c0021aaa2d06740228142c99b87",
           "751c68fe9c8a5a0bd0eacc7b05476280",
           "ed539ac9324b9a767f754c5fba83da44",
           ##################################
           "b33ed6dae7f3f9a47e3e23063ea8b856",
           "f705cee89a80c5fe52765f7bcc150e99",
           "9349b6afe187a8fe8b630b85fa419032",
           "dbc9fc1f7e862fc158690e032b2f207d",
           "1e03e8e903412f6a644a2d223e046374",
           "f92318d19d1187916f8a348e702a2fdd",
           "bbd4e67db9acb506255e1eda683ed2c5",
           "898250b0c66bd859463321942ecd410e",
           "58d447def53b3927935783a5d4802195",
           "c00f0210dec287ad94dfe69e5df7b82f"
           ]


def get_data(type_id, longitude, latitude, key_count):
    num = 0

    for page in range(1, 46):  # 官方限制，只能到45页,只能一页一页请求
        if page == 45:  # 若POI数量超出范围，请考虑POI类型切分
            print("out of range, current page is 45")
        url = url1 + type_id + url2 + longitude + "," + latitude + url3 + str(page) + url4 + key_set[key_count]
        data_poi = requests.get(url, headers={'Connection': 'close'})
        requests.adapters.DEFAULT_RETRIES = 5
        s = requests.session()
        s.keep_alive = False
        ret = data_poi.json()
        if "pois" not in ret:
            print("pois not in message, current key volume is run out of")
            return -1
        aa = ret["pois"]

        if len(aa) == 0:  # 若解析的Json为空，即没有到45页
            break
        for k in range(0, len(aa)):
            num += 1
    return num


data = pd.read_csv("C:/Pywork/IS_Charge/data/d.csv")
long = data['destination_log']
lati = data['destination_lat']
# print(types['poi'][1])

keyid = 12
print(len(lati))
for i in range(16879, 17000):
    pois = []
    for j in range(13):
        type_num = get_data(types['poi'][j], str(long[i]), str(lati[i]), keyid)
        if type_num == -1:
            keyid += 1
            j -= 1
        else:
            pois.append(type_num)
    print("id：" + str(i) + " 所用keyid：" + str(keyid))
    print(pois)
    c = pd.DataFrame([pois])
    c.to_csv("C:/Users/Administrator/Desktop/17000.csv",
             encoding='utf-8', header=None, index=None, mode='a')
# 20231
