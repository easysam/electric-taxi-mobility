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


def load2drop_matrix(transactions, load_clusters, drop_clusters):
    logging.info('load2drop_matrix')
    # 统计转移矩阵
    transits = []
    transits_time_duration = []
    duration = timedelta(minutes=20)
    for interval in range(int(timedelta(days=1) / duration)):
        hour = (interval * duration) // timedelta(hours=1)
        start_min = ((interval * duration) % timedelta(hours=1)).seconds / 60
        end_min = ((interval * duration) % timedelta(hours=1) + duration).seconds / 60
        logging.info('processing %d\'th interval' % interval)
        transit = np.zeros((len(load_clusters), len(drop_clusters)))
        transit_time_duration = np.zeros((len(load_clusters), len(drop_clusters)))
        for i, k in enumerate(load_clusters):
            all_trans = transactions.loc[(transactions['load_label'] == i) &
                                         (transactions['begin_time'].dt.hour == hour) &
                                         (transactions['begin_time'].dt.minute > start_min) &
                                         (transactions['begin_time'].dt.minute < end_min)]
            for j, l in enumerate(drop_clusters):
                temp_od = all_trans.loc[(all_trans['destination_cube'].apply(lambda _cube: _cube in l['hotpots']))]
                transit[i, j] = len(temp_od.index)
                temp_time_duration = temp_od['end_time'] - temp_od['begin_time']
                transit_time_duration[i, j] = temp_time_duration.dt.total_seconds().mean()
                # print(temp_time_duration.dt.total_seconds().describe())
            transit[i] /= len(all_trans.index)
        transits.append(transit)
        transits_time_duration.append(transit_time_duration)
    return transits, transits_time_duration


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
    path = r'data/transaction_201407.csv'
    transactions = pd.read_csv(path, parse_dates=['begin_time', 'end_time'], infer_datetime_format=True, low_memory=False)


    #transactions = data_loader.load_od(scale='full', common=False)
    common = data_loader.load_trajectory_od_intersection() #电车的id

    # 筛选出处于bbox中的points
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

    # 聚类前绘图
    # plot_points(filtered_od, coordinates='geodetic', hour=-1)
    # plot_points(filtered_od, coordinates='cube', hour=-1)

    # 聚类
    logging.info('Clustering load event')
    start_time = time.time()
    load_clusters = hotspots_discovery_meanshift(filtered_od, event='load')
    print("--- Clustering using %s seconds ---" % (time.time() - start_time))

    logging.info('Clustering drop event')
    start_time = time.time()
    drop_clusters = hotspots_discovery_meanshift(filtered_od, event='drop')
    print("--- Clustering using %s seconds ---" % (time.time() - start_time))

    # After merge, records out of bbox and not in common is null in 8 tail columns
    df_to_dump = pd.merge(transactions, filter_od[['Licence', 'begin_time', 'original_x', 'original_y',
                                                   'original_cube', 'destination_x', 'destination_y',
                                                   'destination_cube', 'load_label', 'drop_label']],
                          on=['Licence', 'begin_time'], how='left', indicator=True)
    print(transactions.shape, df_to_dump.shape)

    df_to_dump.to_csv(
        os.path.join(r'data/od/', 'full_od_with_hotpots_v4.csv'),
        index=False)

    # Add geodetic coordinates to hot spots
    arranged_load = [{'x': item['x'], 'y': item['y'],
                      'longitude': cube_to_coordinate(item['cube'], to_geodetic=True)[0],
                      'latitude': cube_to_coordinate(item['cube'], to_geodetic=True)[1],
                      'cube': item['cube'], 'hotpots': list(set(item['hotpots'])),
                      'cube_geodetic': [cube_to_coordinate(cube, to_geodetic=True) for cube in
                                        list(set(item['hotpots']))]
                      } for item in load_clusters]

    arranged_drop = [{'x': item['x'], 'y': item['y'],
                      'longitude': cube_to_coordinate(item['cube'], to_geodetic=True)[0],
                      'latitude': cube_to_coordinate(item['cube'], to_geodetic=True)[1],
                      'cube': item['cube'], 'hotpots': list(set(item['hotpots'])),
                      'cube_geodetic': [cube_to_coordinate(cube, to_geodetic=True) for cube in
                                        list(set(item['hotpots']))]
                      } for item in drop_clusters]
    # Add hot-spots id to detail info
    for i in range(len(arranged_load)):
        arranged_load[i]['id'] = i

    for i in range(len(arranged_drop)):
        arranged_drop[i]['id'] = i

    logging.info('Dump load and drop clusters')
    with open('data/transit_matrix/full_load_clusters.list_of_dict_v4', 'wb') as f:
        pickle.dump(arranged_load, f)
    with open('data/transit_matrix/full_drop_clusters.list_of_dict_v4', 'wb') as f:
        pickle.dump(arranged_drop, f)

    exit(0)
    # 统计转移矩阵
    transits, transits_time_duration = load2drop_matrix(filtered_od, load_clusters, drop_clusters)
    # 保存转移矩阵
    with open('data/transit_matrix/l2d_v3.list_of_numpy', 'wb') as f:
        pickle.dump(transits, f)
    with open('data/transit_matrix/l2d_time_v3.list_of_numpy', 'wb') as f:
        pickle.dump(transits_time_duration, f)
