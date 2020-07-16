import pandas as pd
import utils.data_loader as data_loader
import utils.display as display
import json
import datetime

if __name__ == '__main__':
    cs_info, _ = data_loader.load_cs(date=datetime.datetime(2014, 7, 1))

    with open('cs_info.json', 'w', encoding='utf_8') as f:
        cs_info.drop(['ID', 'Online'], axis=1).to_json(f)
