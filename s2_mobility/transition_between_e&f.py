#coding=gbk
import os
import yaml
import datetime
import utils.data_loader as data_loader
import s1_preprocessing.hotspot.hotpots_discovery_utils as hotspot
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from pandas import Series

def cube():
    m = 400
    n = 800
    bl_lng = 113.764635
    bl_lat = 22.454727
    tr_lng = 114.608972
    tr_lat = 22.842654
    X = (tr_lng - bl_lng) / n
    Y = (tr_lat - bl_lat) / m

    dir = r"./data/transit_matrix"
    file = open(os.path.join(dir, 'all_od_cube.csv'), 'w+')
    for i in range(0,m):
        lat=(tr_lat-Y/2-Y*i)
        for j in range(0,n):
            log=bl_lng+X/2+X*j
            id=j+i*n
            file.write(str(lat)+","+str(log)+","+str(id)+"\n")
    file.close()

    p_hs = pd.read_csv('./data/transit_matrix/all_od_cube.csv')

    load_hotspots_loc = p_hs[['cube']].values
    print(load_hotspots_loc)
    print(load_hotspots_loc.__len__())


def p2d_ef():
    ##################################################################################################################
    # Statistically count transit from pick-up to drop-off
    od_with_hs = data_loader.load_od(common=False)
    od_with_hs=hotspot.generate_cube_index(od_with_hs)
    od_with_hs = od_with_hs.loc[:, ['Licence', 'begin_time', 'end_time', 'original_cube', 'destination_cube']].copy()
    ev = data_loader.load_trajectory_od_intersection()

    od_with_hs_ev=od_with_hs[od_with_hs['Licence'].isin(ev)]
    od_with_hs_fv=od_with_hs[~(od_with_hs['Licence'].isin(ev))]

    print("*")

    _transits_prob, _transits_mean_time = count_transits(od_with_hs_ev, od_with_hs_fv, 'original_cube', 'destination_cube', 'begin_time',
                                                         'end_time')
    _transits_prob2, _transits_mean_time2 = count_transits(od_with_hs_fv, od_with_hs_ev, 'original_cube',
                                                         'destination_cube', 'begin_time',
                                                         'end_time')
    print(coeff(_transits_prob,_transits_prob2))
    print(coeff(_transits_mean_time,_transits_mean_time2))
    return _transits_prob, _transits_mean_time

def count_transits(df_od,df_od2, a, b, a_t, b_t):
    _transits_prob = []
    _transits_mean_time = []
    start_time = datetime.datetime(1, 1, 1, 0, 0, 0)
    duration = datetime.timedelta(minutes=20)

    # Calculate time in the trip
    df_od['time_duration'] = (df_od[b_t] - df_od[a_t]).dt.total_seconds()
    # Extract time of day
    df_od['time_of_day'] = df_od[a_t].dt.time
    df_od.sort_values('time_of_day', axis=0, inplace=True)

    df_od2['time_duration'] = (df_od2[b_t] - df_od2[a_t]).dt.total_seconds()
    # Extract time of day
    df_od2['time_of_day'] = df_od2[a_t].dt.time
    df_od2.sort_values('time_of_day', axis=0, inplace=True)


    for interval in range(int(datetime.timedelta(days=1) / duration)):
        begin_time = start_time + interval * duration
        print(begin_time)
        end_time = begin_time + duration - datetime.timedelta(seconds=1)
        begin_time, end_time = begin_time.time(), end_time.time()
        begin_index = df_od['time_of_day'].searchsorted(begin_time, side='left')
        end_index = df_od['time_of_day'].searchsorted(end_time, side='right')

        begin_index2 = df_od2['time_of_day'].searchsorted(begin_time, side='left')
        end_index2 = df_od2['time_of_day'].searchsorted(end_time, side='right')

        interval_od = df_od.iloc[begin_index: end_index] #part
        interval_od2=df_od2.iloc[begin_index2: end_index2] #all
        print("**")

        transit = same_size_matrix(interval_od, interval_od2, a, b ,1)
        # Delete noise related data, which represented by -1
        #transit.drop(-1, axis=0, inplace=True)
        #transit.drop(-1, axis=1, inplace=True)

        transit_time = same_size_matrix(interval_od, interval_od2, a, b ,2)

        # Delete noise related data, which represented by -1
        #transit_time.drop(-1, axis=0, inplace=True)
        #transit_time.drop(-1, axis=1, inplace=True)

        #transit_time = transit_time.loc[~transit.isna().all(axis=1)]
        #transit = transit.loc[~transit.isna().all(axis=1)]
        _transits_prob.append(transit)
        _transits_mean_time.append(transit_time)

    return _transits_prob, _transits_mean_time

def same_size_matrix(df,df2,a,b,mode):
    '''
    make the shape of the transfer matrix calculated from the random data is the same as that obtained from the complete data
    :param df: part of data
    :param df2: complete data
    :param a: transition origin
    :param b: transition destination
    :return: 
    '''
    print("***")
    if mode == 1:
        newdf = df.groupby([a, b]).size()
        newdf2 = df2.groupby([a, b]).size()
    else:
        newdf = df.groupby([a, b])['time_duration'].mean()
        newdf2 = df2.groupby([a, b])['time_duration'].mean()

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

    series_all = Series(len(land)*[0], index=[ind,col])
    series_all[l1]=newdf
    finaltransit=series_all.unstack(level=-1)

    l = list(set(ind).difference(set(indexlist)))

    p_hs = pd.read_csv('./data/transit_matrix/all_od_cube.csv')

    a = p_hs[p_hs['id'].isin(l)][['lat', 'log']].values
    a_list = p_hs[p_hs['id'].isin(l)].index.tolist()

    p_hs_others = p_hs[p_hs['id'].isin(indexlist)]
    others = p_hs_others[['lat', 'log']].values
    nearest_offset = haversine_distances(np.radians(a), np.radians(others)).argmin(axis=1)
    load_hs = p_hs.loc[p_hs['id'].isin(transit.index), 'id'].iloc[nearest_offset]

    finaltransit.loc[a_list]=finaltransit.loc[load_hs].values.tolist()
    #print(finaltransit.loc[load_hs].values.tolist())

    return finaltransit



def coeff(df,df_part):
    # Calculated correlation coefficient
    s = []
    s2 = []
    for i in range(0, 72):
        a = df[i].fillna(0).values
        # a=np.array(p2d_transits_prob[i], dtype = 'float64')
        # a=a.astype('float64')
        a = a.flatten()
        al = a.tolist()
        s = s + al

        b = df_part[i].fillna(0).values
        # b= b.astype('float64')
        b = b.flatten()
        bl = b.tolist()
        s2 = s2 + bl
    pccs = pearsonr(np.array(s), np.array(s2))
    return pccs[0]

if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])
    p2d_ef()
    #cube()