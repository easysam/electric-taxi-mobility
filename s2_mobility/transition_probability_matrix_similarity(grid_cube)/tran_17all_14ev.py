import os
import pickle
import yaml
from tqdm import tqdm
import datetime
import utils.data_loader as data_loader
import s1_preprocessing.hotspot.hotpots_discovery_utils as hotspot
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from pandas import Series
import random

def coeff(df,df_part):
    # Calculated correlation coefficient
    s = []
    s2 = []
    for i in range(0, 24):
        a = df[i].fillna(0).values
        a = a.flatten()
        al = a.tolist()
        s = s + al

        b = df_part[i].fillna(0).values
        b = b.flatten()
        bl = b.tolist()
        s2 = s2 + bl

    pccs = pearsonr(np.array(s), np.array(s2))
    return pccs[0]

def same_size_matrix(df,df2,a,b):
    '''
    make the shape of the transfer matrix calculated from the random data is the same as that obtained from the complete data
    :param df: part of data
    :param df2: complete data
    :param a: transition origin
    :param b: transition destination
    :return: 
    '''
    newdf = df.groupby([a, b]).size()
    newdf2 = df2.groupby([a, b]).size()

    l1 = newdf.index.tolist()
    l2 = newdf2.index.tolist()
    land = list(set(l1).union(set(l2)))

    ind = []
    col = []
    for i in range(0, len(land)):
        ind.append(land[i][0])
        col.append(land[i][1])

    transit = newdf.unstack(level=-1)
    indexlist = transit.index.tolist()

    series_all = Series(len(land) * [0], index=[ind, col])
    series_all[l1] = newdf
    finaltransit = series_all.unstack(level=-1)

    # blank load_label need to be filled in
    l = list(set(ind).difference(set(indexlist)))

    p_hs = pd.read_csv('./data/transit_matrix/all_od_cube.csv')

    a = p_hs[p_hs['id'].isin(l)][['lat', 'log']].values
    a_list = p_hs[p_hs['id'].isin(l)].index.tolist()

    p_hs_others = p_hs[p_hs['id'].isin(indexlist)]
    others = p_hs_others[['lat', 'log']].values
    nearest_offset = haversine_distances(np.radians(a), np.radians(others)).argmin(axis=1)
    load_hs = p_hs.loc[p_hs['id'].isin(transit.index), 'id'].iloc[nearest_offset]

    finaltransit.loc[a_list] = finaltransit.loc[load_hs].values.tolist()

    return finaltransit


def count_transits(df_od,df_od2, a, b, a_t, b_t):
    '''
    Statistically compute the transit probability 
    :param df_od: part of od data
    :param df_od2: all od data 
    :param a: transition origin
    :param b: transition destination
    :param a_t: 
    :param b_t: 
    :return: transit probability for all od & part of od
    '''
    _transits_prob = []
    _transits_prob_part = []

    start_time = datetime.datetime(1, 1, 1, 0, 0, 0)
    duration = datetime.timedelta(minutes=60)

    # Extract time of day
    df_od['time_of_day'] = df_od[a_t].dt.time
    df_od.sort_values('time_of_day', axis=0, inplace=True)

    # Extract time of day
    df_od2['time_of_day'] = df_od2[a_t].dt.time
    df_od2.sort_values('time_of_day', axis=0, inplace=True)

    for interval in range(int(datetime.timedelta(days=1) / duration)):
        begin_time = start_time + interval * duration
        end_time = begin_time + duration - datetime.timedelta(seconds=1)
        begin_time, end_time = begin_time.time(), end_time.time()
        print(begin_time)

        begin_index = df_od['time_of_day'].searchsorted(begin_time, side='left')
        end_index = df_od['time_of_day'].searchsorted(end_time, side='right')

        begin_index2 = df_od2['time_of_day'].searchsorted(begin_time, side='left')
        end_index2 = df_od2['time_of_day'].searchsorted(end_time, side='right')

        interval_od = df_od.iloc[begin_index: end_index]
        interval_od2=df_od2.iloc[begin_index2: end_index2]

        transit = same_size_matrix(interval_od, interval_od2, a, b )
        print(transit)
        transit2 = same_size_matrix(interval_od2, interval_od, a, b )

        _transits_prob_part.append(transit)
        _transits_prob.append(transit2)

    return _transits_prob,_transits_prob_part

def p2d_part():
    ##################################################################################################################
    # Statistically count transit from pick-up to drop-off
    #17-all
    path = r'data/od/17_et_od_with_cubes.csv'
    transactions = pd.read_csv(path, parse_dates=['begin_time', 'end_time'], infer_datetime_format=True, low_memory=False)
    od_with_hs = transactions.loc[:, ['id', 'begin_time', 'end_time', 'original_cube', 'destination_cube']].copy()
    path = r'data/201706et_list.csv'
    ef17 = pd.read_csv(path, infer_datetime_format=True, low_memory=False)
    ev17 = ef17[ef17['is_Ecar'] == True]['license']
    ef17_part = random.sample(ev17.values.tolist(), 645)
    od_with_hs = od_with_hs[od_with_hs['id'].isin(ef17_part)]

    #14ev
    ev_14 = data_loader.load_trajectory_od_intersection()
    path=r'data/od/full_od_with_cube.csv'
    od_with_hs2 = pd.read_csv(path, parse_dates=['begin_time', 'end_time'], infer_datetime_format=True, low_memory=False)
    #od_with_hs2 = od_with_hs2[od_with_hs2['Licence'].isin(ev_14)]
    od_with_hs2 = od_with_hs2.loc[:, ['Licence', 'begin_time', 'end_time', 'original_cube', 'destination_cube']].copy()

    _transits_prob,_transits_prob2 = count_transits(od_with_hs2, od_with_hs, 'original_cube', 'destination_cube',
                                                               'begin_time', 'end_time')
    print(coeff(_transits_prob,_transits_prob2))
    return _transits_prob

if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])
    ##################################################################################################################
    # Statistically count transit from pick-up to drop-off
    p2d_transits_prob2= p2d_part()

