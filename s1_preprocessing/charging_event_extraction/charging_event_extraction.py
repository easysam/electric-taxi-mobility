import os
import yaml
import datetime
import logging
import numpy as np
import pandas as pd
from  tqdm import tqdm
from sklearn.metrics.pairwise import haversine_distances
from utils import data_loader, display
from utils.vector_haversine_distances import haversine_np


# 统计带标注数据集中每个站的充电事件轨迹点离站的距离
def show_max_distance_of_ce(ce):
    return df_trajectory.loc[df_trajectory.index[ce['start']]: df_trajectory.index[ce['end']], 'distance_to_cs'].max()


def show_max_distance_after_arrival(ce):
    trajectory_location = df_trajectory.loc[ce['arrived']: ce['end'], ['Latitude', 'Longitude']].values
    cs_location = cs_info.loc[cs_info['cs_name'] == ce['cs'], ['Latitude', 'Longitude']].values

    # earth radius(km)
    AVG_EARTH_RADIUS = 6371.0088
    # calculate distance between drop location and CS location, and midian, min, max, mean
    distances_to_cs = haversine_distances(np.radians(trajectory_location), np.radians(cs_location)) * AVG_EARTH_RADIUS

    return distances_to_cs.max()


def charging_time(ce):
    start_time = df_trajectory.at[df_trajectory.index[ce['start']], 'timestamp']
    end_time = df_trajectory.at[df_trajectory.index[ce['end']], 'timestamp']
    return end_time - start_time


def show_max_distance_of_station(df):
    return pd.Series({'max_distance': df.apply(show_max_distance_of_ce, axis=1).max(),
                      'max_arrival_distance': df.apply(show_max_distance_after_arrival, axis=1).max(),
                      'mean_charging_time': df.apply(charging_time, axis=1).mean(),
                      'min_charging_time': df.apply(charging_time, axis=1).min(),
                      'max_charging_time': df.apply(charging_time, axis=1).max()})


def find_potential_ce(traj_df, with_status=True):
    logging.info('Finding pce')

    traj_df = traj_df.merge(cs_ce_info['max_distance'] * 1.1, left_on='cs_name', right_index=True, how='left')

    traj_df['in_range'] = traj_df.distance_to_cs < traj_df.max_distance

    traj_df['grp'] = ((traj_df.status != traj_df.status.shift())
                      | (traj_df.Licence != traj_df.Licence.shift())
                      | (traj_df.in_range != traj_df.in_range.shift())
                      | (traj_df['in_range'] & (traj_df['cs_name'] != traj_df['cs_name'].shift()))).cumsum()

    df_pce = pd.DataFrame({
        'licence': traj_df.groupby('grp')['Licence'].first(),
        'begin_time': traj_df.groupby('grp')['timestamp'].first(),
        'begin_time_index': traj_df.groupby('grp').apply(lambda x: x.index[0]),
        'end_time': traj_df.groupby('grp')['timestamp'].last(),
        'end_time_index': traj_df.groupby('grp').apply(lambda x: x.index[-1]),
        'charging_duration': traj_df.groupby('grp')['timestamp'].last() - traj_df.groupby('grp')['timestamp'].first(),
        'consecutive': traj_df.groupby('grp').size(),
        'grp': traj_df.groupby('grp')['grp'].first(),
        'cs_name': traj_df.groupby('grp')['cs_name'].first(),
        'in_range': traj_df.groupby('grp')['in_range'].first(),
        'status': traj_df.groupby('grp')['status'].first(),
        'valid': traj_df.groupby('grp')['valid'].all(),
    }).reset_index(drop=True)

    return df_pce


def filter_pce(traj_df, t_df_pce, with_status=True):
    logging.info('Filtering pce')

    # 给出pce是否为ce的标记
    t_df_pce['ce'] = (t_df_pce['in_range']
                      & (~t_df_pce['status'])
                      & (t_df_pce['charging_duration'] > datetime.timedelta(minutes=30))
                      )

    t_df_pce['valid'] = t_df_pce['ce'] | t_df_pce['valid']

    t_df_pce['grp_end_flag'] = (t_df_pce['licence'] != t_df_pce['licence'].shift(-1))
    t_df_pce.loc[t_df_pce['ce'] == True, 'grp_end_flag'] = True

    t_df_pce['with_pre_grp'] = t_df_pce.loc[::-1, 'grp_end_flag'].cumsum()[::-1]

    df_result = pd.DataFrame({
        'licence': t_df_pce.groupby('with_pre_grp', sort=False)['licence'].last(),
        'begin_time': t_df_pce.groupby('with_pre_grp', sort=False)['begin_time'].last(),
        'begin_time_index': t_df_pce.groupby('with_pre_grp', sort=False)['begin_time_index'].last().astype(int),
        'end_time': t_df_pce.groupby('with_pre_grp', sort=False)['end_time'].last(),
        'end_time_index': t_df_pce.groupby('with_pre_grp', sort=False)['end_time_index'].last().astype(int),
        'charging_duration': t_df_pce.groupby('with_pre_grp', sort=False)['charging_duration'].last(),
        'grp': t_df_pce.groupby('with_pre_grp', sort=False)['grp'].last(),
        'ce': t_df_pce.groupby('with_pre_grp', sort=False)['ce'].last(),
        'cs_name': t_df_pce.groupby('with_pre_grp', sort=False)['cs_name'].last(),
        'valid': t_df_pce.groupby('with_pre_grp', sort=False)['valid'].all(),
    })

    df_result = df_result.loc[df_result['ce']]
    df_result.drop(['ce', 'grp'], axis=1, inplace=True)
    df_result.reset_index(drop=True, inplace=True)
    return df_result


def search_arrival(ce):
    #
    # begin: based on threshold, start: start charge, arrival: arrival CS
    #
    ce_trajectory = df_trajectory.loc[ce['begin_time_index']: ce['end_time_index']]
    if ce['charging_duration'] > datetime.timedelta(hours=1.5):
        end_time_index = ce_trajectory['timestamp'].searchsorted(ce['begin_time'] + datetime.timedelta(hours=1.5))
        end_time_index = ce_trajectory.index[end_time_index]
        ce_trajectory = df_trajectory.loc[ce['begin_time_index']: end_time_index]
    stay_points = ce_trajectory.groupby(['Longitude', 'Latitude'])['interval'].sum().sort_values(ascending=False)
    ce_logitude, ce_latitude = stay_points.index[0]
    # 第一个在充电点速度为0的点
    try:
        # 取第一个在充电点速度为0的点的值索引
        start_charging_index = ce_trajectory.loc[(ce_trajectory['Longitude'] == ce_logitude)
                                                 & (ce_trajectory['Latitude'] == ce_latitude)
                                                 & ~ce_trajectory['Speed']].index[0]
        try:
            # 取开始充电时间，若等待时间>10分钟，将值索引-1，为了应对开始充电点距上一轨迹点时间间隔过长的问题
            start_charging = ce_trajectory.at[start_charging_index - 1, 'timestamp']
            if ce_trajectory.at[start_charging_index, 'timestamp'] - start_charging < datetime.timedelta(minutes=10):
                start_charging = ce_trajectory.at[start_charging_index, 'timestamp']
        except KeyError:
            # 值索引-1可能会导致keyerror，若导致，则不-1
            start_charging = ce_trajectory.at[start_charging_index, 'timestamp']
    except IndexError:
        # 可能不存在在充电点速度为0的点
        start_charging = ce['begin_time']
    #     if (ce['licence'] == '粤B0BA49') and (ce['cs_name'] == 'A08'):
    #         print('max_arrival_distance:', ce['max_arrival_distance'])
    #         print(stay_points)
    #         print(start_charging_index, start_charging)
    #         print(ce_trajectory)

    # 计算含有等待的轨迹点离CS的距离
    wait_contained_trajectory = df_trajectory.loc[ce['begin_time_index']: ce['begin_time_index'] - 100: -1].copy()
    wait_location = wait_contained_trajectory[['Latitude', 'Longitude']].values
    cs_location = cs_info.loc[cs_info['cs_name'] == ce['cs_name'], ['Latitude', 'Longitude']].values
    # earth radius(km)
    AVG_EARTH_RADIUS = 6371.0088
    # calculate distance between drop location and CS location, and midian, min, max, mean
    distances_to_cs = haversine_distances(np.radians(wait_location), np.radians(cs_location)) * AVG_EARTH_RADIUS
    wait_contained_trajectory['distance_to_cs'] = distances_to_cs

    # 如果充电点的累计驻留时间小于50分钟，则认为轨迹乱跳，不专门检测开始充电时间点
    if stay_points.iloc[0] < datetime.timedelta(minutes=30):
        start_charging = ce_trajectory.at[ce['begin_time_index'], 'timestamp']

    # 取arrival_index
    outer_points = wait_contained_trajectory.loc[
        wait_contained_trajectory['distance_to_cs'] > ce['max_arrival_distance']]
    if len(outer_points.index) == 0:
        #         print(ce)
        arrival_index = ce['begin_time_index']
    else:
        arrival_index = outer_points.index[0]
        arrival_index = arrival_index if arrival_index == ce['begin_time_index'] else arrival_index + 1

    arrival_time = df_trajectory.at[arrival_index, 'timestamp']

    # 计算等待时间并赋值给 充电事件 ce['waiting_duration']
    if datetime.timedelta(seconds=0) < start_charging - arrival_time:
        ce['arrival_time'] = arrival_time
        ce['start_charging'] = start_charging
        waiting_duration = start_charging - arrival_time
    else:
        ce['arrival_time'] = ce['begin_time']
        ce['start_charging'] = ce['begin_time']
        waiting_duration = datetime.timedelta(seconds=0)

    ce['waiting_duration'] = waiting_duration
    ce['charging_duration'] = ce['end_time'] - ce['start_charging']
    #     if (ce['licence'] == '粤B0BA49') and (ce['cs_name'] == 'A08'):
    #         print(arrival_time, start_charging)
    #         print(ce)

    return ce


if __name__ == '__main__':
    # Initialization
    tqdm.pandas()
    display.configure_pandas()
    display.configure_logging()
    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    # Load raw trajectory data
    df_trajectory = data_loader.load_trajectory(with_status=True)
    df_trajectory.drop(['cs_index', 'cs_lat', 'cs_log', 'cs_points', 'cs_date'], axis=1, inplace=True)
    cs_info, date = data_loader.load_cs(date=datetime.datetime(2014, 7, 1))
    gt = pd.read_csv('./gt.csv', sep='\s*,\s*', engine='python', encoding='utf-8-sig')

    # 计算每个轨迹点距离其前面一个轨迹点的球面距离
    logging.info('Calculating distance between points.')
    df_trajectory['Distance'] = haversine_np(df_trajectory.Longitude.shift(), df_trajectory.Latitude.shift(),
                                             df_trajectory['Longitude'], df_trajectory['Latitude'])
    df_trajectory.loc[df_trajectory['Licence'] != df_trajectory['Licence'].shift(), 'Distance'] = None
    # 计算时间间隔
    df_trajectory['interval'] = df_trajectory['timestamp'] - df_trajectory['timestamp'].shift()
    df_trajectory.loc[df_trajectory['Licence'] != df_trajectory['Licence'].shift(), 'interval'] = None
    # 根据阈值给出大间隔标记
    df_trajectory['big_interval'] = df_trajectory['interval'] > datetime.timedelta(minutes=15)
    # 大间隔且点距超过0.1KM的定为异常点
    df_trajectory['valid'] = ~(df_trajectory['big_interval'] & (df_trajectory['Distance'] > 0.1))

    # Extract charging event info (charger distance to center location etc.) for each charging station
    cs_ce_info = gt.groupby('cs').apply(show_max_distance_of_station).sort_values('max_distance', ascending=False)

    # 找到潜在ce
    df_pce = find_potential_ce(df_trajectory)
    # 过滤pce
    result = filter_pce(df_trajectory, df_pce)

    result = result.merge(cs_ce_info['max_arrival_distance'] * 1.1, left_on='cs_name', right_index=True, how='left')

    # Search arrival time for each charging event, to calculate the queuing time duration length
    waiting_duration = result.progress_apply(search_arrival, axis=1)

    # todo: one of 'arrival_time', 'start_charging', 'begin_time' is useless, which needs to be inferred in above code.
    waiting_duration[['licence', 'arrival_time', 'start_charging', 'begin_time', 'end_time', 'waiting_duration',
                      'charging_duration', 'cs_name', 'valid']].to_csv('data/charging_event/ce_30min.csv', index=False)
