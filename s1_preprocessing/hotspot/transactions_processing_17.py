#coding=gbk
import time
import yaml
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from s1_preprocessing.hotspot.hotpots_discovery_utils import plot_points, generate_cube_index, hotspots_discovery_meanshift, \
    hotspots_by_dbscan, generate_data_for_elki, cube_to_coordinate
import logging
import pickle
import utils.data_loader as data_loader
import utils.display as display
import os
from tqdm import tqdm

tqdm.pandas()

if __name__ == '__main__':

    # configure the working directory to the project root path
    with open("../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])
    ##################################################################################################################

    # initialization
    display.configure_pandas()
    display.configure_logging()


    # 读取transactions数据
    path = r'data/od/201706_et_od.csv'
    transactions = pd.read_csv(path, parse_dates=['begin_time', 'end_time'], infer_datetime_format=True, low_memory=False)
    transactions['in_bbox'] = ((113.764635 < transactions['destination_log'])
                               & (transactions['destination_log'] < 114.608972)
                               & (22.454727 < transactions['destination_lat'])
                               & (transactions['destination_lat'] < 22.842654)
                               & (113.764635 < transactions['original_log'])
                               & (transactions['original_log'] < 114.608972)
                               & (22.454727 < transactions['original_lat'])
                               & (transactions['original_lat'] < 22.842654))
    filter_od = transactions.loc[transactions.in_bbox].reset_index(drop=True)
    print('Shape of transactions that out of bbox:', transactions.shape[0] - filter_od.shape[0])
    # 将points分到cubes里
    filtered_od = generate_cube_index(filter_od)

    # 读hotspot文件
    path = 'data/transit_matrix/full_load_clusters.list_of_dict_v4'
    with open(path, 'rb') as f:
        load_clusters = pickle.load(f)

    path2 = 'data/transit_matrix/full_drop_clusters.list_of_dict_v4'
    with open(path2, 'rb') as f2:
        drop_clusters = pickle.load(f2)

    # 为每一行交易数据找匹配的热区
    def find_load_cluster(df,load_c=None):
        load_label=-1
        for i in load_c:
            if df in i['hotpots']:
                load_label = i['id']
                break
        return load_label

    filtered_od['load_label'] = filtered_od['original_cube'].progress_apply(find_load_cluster,load_c = load_clusters)
    filtered_od['drop_label'] = filtered_od['destination_cube'].progress_apply(find_load_cluster,load_c = drop_clusters)
    print(filtered_od)

    filtered_od.to_csv(os.path.join(r'data/od/', '17_et_od_with_hotpots.csv'),index=False)





