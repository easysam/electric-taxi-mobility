import os
import datetime
import numpy as np
import pandas as pd
import logging
import yaml
from tqdm import tqdm
from utils import display
from utils import data_loader
from utils.vector_haversine_distances import haversine_np
from sklearn.metrics.pairwise import haversine_distances


def fetch_departure_info(ce_record, df_od):
    begin = df_od['id'].searchsorted(ce_record['id'])
    end = df_od['id'].searchsorted(ce_record['id'], side='right')
    od_of_id_in_ce = df_od.iloc[begin:end]
    offset = od_of_id_in_ce['d_t'].searchsorted(ce_record['start_charging']) - 1
    # 如果不存在早于该ce的od（loc<0），则返回空值
    if offset < 0:
        return pd.Series(index=['d_t', 'd_lng', 'd_lat', 'd_l'],)
    else:
        return od_of_id_in_ce.loc[od_of_id_in_ce.index[offset], ['d_t', 'd_lng', 'd_lat', 'd_l']]


def find_traveled_distance(od, df_trajectory, df_ce):
    # 本函数计算该od结束时，距上次充电结束，已行驶的距离

    # 从df_ce中查找该od上次充电结束的时间
    begin = df_ce['id'].searchsorted(od['id'])
    end = df_ce['id'].searchsorted(od['id'], side='right')
    # 如果找不到该车牌，即为begin==end，返回空，即找不到该od之前的充电记录
    if begin == end:
        # 特殊退出！！！！！！
        return pd.Series({'traveled_after_charged': None, 'to_charge': None, 'seeking_duration': None})
    licence_ce = df_ce.iloc[begin: end]

    # 该分支判断给出该od结束后是否去充电了
    seeking_duration = None
    if od['d_t'].to_datetime64() in licence_ce['last_d_t'].unique():
        charged = True
        # calculate seeking time
        ce_index = licence_ce['last_d_t'].searchsorted(od['d_t'])
        seeking_duration = licence_ce.at[licence_ce.index[ce_index], 'start_charging'] - od['d_t']
    else:
        charged = False

    # 得到该od上次充电结束的时间所在的ce的index（该index为值index，而非偏移index）
    travel_begin_index_in_ce = begin + licence_ce['end_time'].searchsorted(od['d_t']) - 1
    # 如果该时间之前还没有充电事件，即为travel_begin_index_in_ce<0，返回空
    if travel_begin_index_in_ce < begin:
        # 特殊退出！！！！！！
        return pd.Series({'traveled_after_charged': None, 'to_charge': charged, 'seeking_duration': seeking_duration})
    # 得到该od上次充电结束的时间
    travel_begin_time = licence_ce.at[travel_begin_index_in_ce, 'end_time']

    # 计算已经行驶的路程，在轨迹df中计算
    begin = df_trajectory['id'].searchsorted(od['id'])
    end = df_trajectory['id'].searchsorted(od['id'], side='right')
    licence_trajectories = df_trajectory[begin: end]
    begin_index = begin + licence_trajectories['ts'].searchsorted(travel_begin_time, side='right')
    end_index = begin + licence_trajectories['ts'].searchsorted(od['d_t'])

    return pd.Series({'traveled_after_charged': licence_trajectories.loc[begin_index: end_index, 'dis'].sum(),
                      'to_charge': charged,
                      'seeking_duration': seeking_duration if charged else None})


if __name__ == '__main__':
    tqdm.pandas()
    display.configure_pandas()
    display.configure_logging()

    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    # load data set
    ce = data_loader.load_ce(version='v5_30min')
    od = data_loader.load_od(with_hotpots=True, version='v4')
    df_trajectories = data_loader.load_trajectory(with_status=False)
    df_trajectories['Distance'] = haversine_np(df_trajectories.longitude.shift(), df_trajectories.latitude.shift(),
                                               df_trajectories['longitude'], df_trajectories['latitude'])
    df_trajectories.loc[df_trajectories['plate'] != df_trajectories['plate'].shift(), 'Distance'] = None

    ce.rename(columns={'licence': 'id'}, inplace=True)
    od.rename(columns={'Licence': 'id', 'begin_time': 'o_t', 'end_time': 'd_t', 'original_log': 'o_lng',
                       'original_lat': 'o_lat', 'destination_log': 'd_lng', 'destination_lat': 'd_lat',
                       'original_x': 'o_x', 'original_y': 'o_y', 'original_cube': 'o_grid',
                       'destination_x': 'd_x', 'destination_y': 'd_y', 'destination_cube': 'd_grid',
                       'load_label': 'o_l', 'drop_label': 'd_l'}, inplace=True)
    df_trajectories.rename(columns={'plate': 'id', 'longitude': 'lng', 'latitude': 'lat', 'timestamp': 'ts',
                                    'velocity': 'v', 'Distance': 'dis'}, inplace=True)

    # calculate
    departure_info = ce.progress_apply(fetch_departure_info, axis=1, args=(od,))
    departure_info.rename(columns={'d_t': 'last_d_t',
                                   'd_l': 'last_d_l'}, inplace=True)
    ce = pd.concat([ce, departure_info], axis=1)

    od = pd.concat([od, od.progress_apply(find_traveled_distance, args=(df_trajectories, ce,), axis=1)], axis=1)

    # Load CS info
    df_cs, dates = data_loader.load_cs(date=datetime.datetime(2014, 7, 1))

    # Drop fake CS
    df_cs = df_cs.loc[~df_cs['cs_name'].isin(['LJDL', 'E04', 'BN0002', 'F11', 'S1', 'S2', 'F12', 'F13', 'F15'])]

    # select drop location and CS location as two array
    drop_location = od[['d_lat', 'd_lng']].to_numpy()
    cs_location = df_cs[['lat', 'lng']].to_numpy()

    # earth radius(km)
    AVG_EARTH_RADIUS = 6371.0088

    # calculate distance between drop location and CS location, and midian, min, max, mean
    distances_to_cs = haversine_distances(np.radians(drop_location), np.radians(cs_location)) * AVG_EARTH_RADIUS
    df_distances_to_cs = pd.DataFrame(distances_to_cs)
    df_distances_to_cs['mid_dis'] = df_distances_to_cs.median(axis=1)
    df_distances_to_cs['min_dis'] = df_distances_to_cs.min(axis=1)
    df_distances_to_cs['max_dis'] = df_distances_to_cs.max(axis=1)
    df_distances_to_cs['mean_dis'] = df_distances_to_cs.mean(axis=1)

    # concat distance data back to transaction data
    pd.concat([od, df_distances_to_cs], axis=1).to_csv(r'data/od/od_with_traveled_v6.csv', index=False)

    # check column names
    # pd.concat([od, df_distances_to_cs], axis=1).columns

    # check distribution of seeking duration of go to charge
    # (pd.concat([od, df_distances_to_cs], axis=1)
    #  .dropna()['seeking_duration'] / np.timedelta64(1, 'm')).plot.kde(ind=range(0, 100, 10))
