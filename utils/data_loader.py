import pandas as pd
import logging
import glob
import os
import pickle
import config
from utils.cs_info import get_cs_info_by_date

project_path = r'C:\Users\hkrept\PycharmProjects\ElectricVehicleMobility'


def load_ce(scale='full', with_source=False, version=None):
    if 'full' == scale:
        if with_source:
            if version is None:
                path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\ce\ce_with_source_info_v9.csv'
            else:
                path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\ce\ce_with_source_info_' + version + '.csv'
            parse_dates = ['begin_time', 'start_charging', 'end_time', 'source_t']
        else:
            if version is None:
                path = os.path.join(project_path, 'data/ce_v5_30min.csv')
                parse_dates = ['arrival_time', 'start_charging', 'begin_time', 'end_time', 'waiting_duration',
                               'charging_duration']
            else:
                path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\ce\ce_' + version + '.csv'
                parse_dates = ['arrival_time', 'start_charging', 'end_time', 'waiting_duration', 'charging_duration']
                logging.info('Loading ' + path)
                temp_df = pd.read_csv(path, parse_dates=parse_dates)
                return temp_df
    elif 'part' == scale:
        raise NotImplementedError
    else:
        raise NotImplementedError
    logging.info('Loading ' + path)
    temp_df = pd.read_csv(path, parse_dates=parse_dates)
    temp_df['waiting_duration'] = temp_df['waiting_duration'].apply(pd.Timedelta)
    temp_df['charging_duration'] = temp_df['charging_duration'].apply(pd.Timedelta)
    return temp_df


def load_trajectory(with_status=False):
    if with_status:
        path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\trajectory\trajectories_with_dis_status.csv'

        logging.info('Loading ' + path)
        return pd.read_csv(path, sep=',', parse_dates=['timestamp'], infer_datetime_format=True, low_memory=False,
                           usecols=['timestamp', 'Licence', 'Longitude', 'Latitude', 'Speed', 'cs_index', 'cs_lat',
                                    'cs_log', 'cs_name', 'cs_points', 'distance_to_cs', 'cs_date', 'status'],
                           na_values=['nan', '?', 'NaN'], header=0, index_col=None)
    else:
        path = os.path.join(project_path, 'data/history_trajectories.csv')
        logging.info('Loading ' + path)
        return pd.read_csv(path, sep=',', parse_dates=['timestamp'], infer_datetime_format=True, low_memory=False,
                           usecols=['plate', 'longitude', 'latitude', 'timestamp', 'velocity'])



def load_trajectory_od_intersection():
    path = os.path.join(project_path, 'data/201407et_list.pickle')
    with open(path, 'rb') as f:
        traj_od_common = pickle.load(f)
    return traj_od_common


def load_od(scale='full', with_hotpots=False, with_feature=False, with_distance=False, version='v3', common=True):
    if not common:
        path = r'C:\Users\hkrept\PycharmProjects\ElectricVehicleMobility\data\transaction_201407.csv'
        logging.info('Loading ' + path)
        df = pd.read_csv(path, parse_dates=['begin_time', 'end_time'], infer_datetime_format=True, low_memory=False)
        return df
    if 'full' == scale:
        if with_hotpots:
            common = load_trajectory_od_intersection()
            path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\od\full_od_with_hotpots_' + version \
                   + '.csv'
            logging.info('Loading ' + path)
            df = pd.read_csv(path, parse_dates=['begin_time', 'end_time'], infer_datetime_format=True, low_memory=False)
            df.fillna(value={'load_label': -1, 'drop_label': -1}, axis=0, inplace=True)
            df = df.astype({'load_label': int, 'drop_label': int})
            return df
        elif with_feature:
            path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\od\od_with_traveled_v5.csv'
            logging.info('Loading ' + path)
            df = pd.read_csv(path, parse_dates=['begin_time', 'end_time'], infer_datetime_format=True, low_memory=False)
            df['seeking_duration'] = pd.to_timedelta(df['seeking_duration'])
            return df
        elif with_distance:
            path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\od\od_with_distance_between_before.csv'
            logging.info('Loading' + path)
            df = pd.read_csv(path, parse_dates=['begin_time', 'end_time', 'last_drop_time'], infer_datetime_format=True,
                             low_memory=False)
            return df
        else:
            path = os.path.join(project_path, 'data/transaction_common_201407.csv')
    elif 'part' == scale:
        if with_hotpots:
            path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\od\od_with_hotpots.csv'
        else:
            path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\od\part_od_20140704_05.csv'
    else:
        raise NotImplementedError
    logging.info('Loading ' + path)
    df = pd.read_csv(path, parse_dates=['begin_time', 'end_time'], infer_datetime_format=True, low_memory=False)
    return df


def load_cs(scale='full', date=None):
    logging.info('Loading \'C:\\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\ChargingStation\'')
    if 'full' == scale:
        if date is None:
            dates_ = []
            list_ = []
            path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\ChargingStation'
            all_files = glob.glob(os.path.join(path, "*"))
            for f in all_files:
                df_c = pd.read_csv(f, sep=',', names=['ID', 'cs_name', 'Longitude', 'Latitude', 'Online', 'chg_points'],
                                   infer_datetime_format=True, low_memory=False, na_values=['nan', '?', 'NaN'])
                f_datetime = pd.to_datetime(f.replace(path + '\ChargeLocation', ''), format='%Y%m')
                dates_.append(f_datetime)
                df_c['Date'] = f_datetime
                list_.append(df_c)
            df_cs = pd.concat(list_, ignore_index=True)
            df_cs.set_index('Date', inplace=True)
        else:
            path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\ChargingStation\ChargeLocation' + \
                   date.strftime('%Y%m')
            df_cs = pd.read_csv(path, sep=',', names=['ID', 'cs_name', 'Longitude', 'Latitude', 'Online', 'chg_points'],
                                infer_datetime_format=True, low_memory=False, na_values=['nan', '?', 'NaN'])
            dates_ = date
    elif 'part' == scale:
        path = r'C:\Users\hkrept\PycharmProjects\ElectricVehicleMobility\data\ChargingStation\ChargeLocation' + \
               date.strftime('%Y%m')
        df_cs = pd.read_csv(path, sep=',', names=['ID', 'cs_name', 'Longitude', 'Latitude', 'Online', 'chg_points'],
                            infer_datetime_format=True, low_memory=False, na_values=['nan', '?', 'NaN'])
        dates_ = date
    else:
        raise NotImplementedError
    return df_cs, dates_


def load_clusters():
    load_data_path = \
        r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\transit_matrix\full_load_clusters.list_of_dict_v4'
    logging.info('Load ' + load_data_path)
    with open(load_data_path, 'rb') as f:
        load_hotpots = pickle.load(f)
    return load_hotpots


def drop_clusters():
    drop_cluster_path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\transit_matrix\full_drop_clusters.list_of_dict_v4'
    logging.info('Load ' + drop_cluster_path)
    with open(drop_cluster_path, 'rb') as f:
        drop_clusters = pickle.load(f)
    return drop_clusters


def road_shp():
    road_shp_wgs84_path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\SZXshp\road_wgs84.csv'
    logging.info('Load ' + road_shp_wgs84_path)
    road_shp_wgs84 = pd.read_csv(road_shp_wgs84_path)
    return road_shp_wgs84


def load_driver_work_info():
    path = config.driver_start_work_info_path
    driver_work_info = pd.read_csv(path, )
    return driver_work_info


def pickle_load(file=None):
    if 'l2d' == file:
        with open(config.l2d_path, 'rb') as f:
            file = pickle.load(f)
    elif 'l2d_t' == file:
        with open(config.l2d_time_path, 'rb') as f:
            file = pickle.load(f)
    elif 'd2l' == file:
        with open(config.d2l_path, 'rb') as f:
            file = pickle.load(f)
    elif 'd2l_t' == file:
        with open(config.d2l_time_path, 'rb') as f:
            file = pickle.load(f)
    elif 'c2l' == file:
        with open(config.c2l_path, 'rb') as f:
            file = pickle.load(f)
    elif 'if_to_charge' == file:
        with open(config.if_to_charge_path, 'rb') as f:
            model = pickle.load(f)
        with open(config.whether_charge_data_scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logging.info('Load whether charge model and data scaler.')
        return model, scaler
    elif 'where_to_charge' == file:
        with open(config.where_to_charge_path, 'rb') as f:
            file = pickle.load(f)
    else:
        raise NotImplementedError
    return file


def load_rest():
    path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\rest\rest_events.csv'
    rest_events = pd.read_csv(path, parse_dates=['start_time', 'end_time', 'duration'], infer_datetime_format=True)
    return rest_events


def load_generated(version='v1'):
    path = r'C:\Users\hkrep\PycharmProjects\ChargingEventsExtraction\data\rest\generated_rest_event_' + version + '.csv'
    generated_rest = pd.read_csv(path)
    return generated_rest
