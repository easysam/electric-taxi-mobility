# 1. extract feature and ground truth; 2. split the data set.
import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import pairwise
from utils import data_loader, display, vector_haversine_distances as vec_hs_dis, od_utils
from s1_preprocessing.hotspot.hotpots_discovery_utils import generate_cube_index
from s2_mobility.transition import count_transits


def extract_transition_demand(_df_od, _m=100, _n=200, threshold=10):
    _df_od = od_utils.filter_in_bbox(_df_od)  # filter the transactions within the bounding box.
    _df_od = generate_cube_index(_df_od, m=_m, n=_n)
    _demand = _df_od.groupby(['original_cube', 'destination_cube']).size().rename("demand").reset_index()
    _demand = _demand.loc[_demand['demand'] > threshold]
    _demand.reset_index(drop=True, inplace=True)
    return _demand


def build_composite_demands(_demand, _et_demand):
    # merge total demand and ET demand
    _df_demands = pd.merge(_demand, _et_demand, on=['original_cube', 'destination_cube'],
                           suffixes=('_all', '_et')).fillna(0)
    _df_demands['rate'] = _df_demands['demand_et'] / _df_demands['demand_all']
    return _df_demands


def build_final_transitions(_df_od, _df_demands, _m=100, _n=200):
    _df_od['duration'] = (_df_od['end_time'] - _df_od['begin_time']).dt.total_seconds()
    _df_od = generate_cube_index(_df_od, m=_m, n=_n)
    _df_od = _df_od[['original_cube', 'destination_cube', 'original_log', 'original_lat', 'destination_log',
                     'destination_lat', 'duration']].groupby(['original_cube', 'destination_cube']).mean()
    _df_od.reset_index(inplace=True)
    # df_od_pairs are final od set which have both duration and demand (>10) info
    _df_od_pairs = pd.merge(_df_demands[['original_cube', 'destination_cube']], _df_od[
        ['original_cube', 'destination_cube', 'original_log', 'original_lat', 'destination_log', 'destination_lat',
         'duration']], left_on=['original_cube', 'destination_cube'], right_on=['original_cube', 'destination_cube'])
    return _df_od_pairs


def pairwise_distance(df1, coord1, df2, coord2):
    avg_earth_radius = 6371.0088
    loc1, loc2 = df1[coord1].to_numpy(), df2[coord2].to_numpy()
    _dis = pairwise.haversine_distances(np.radians(loc1), np.radians(loc2)) * avg_earth_radius
    return _dis


def feature_extraction(_od, _cs, _neighbor_num=3):
    # distances from end point to charging station preparation
    arr_origin_dis_to_cs = pairwise_distance(_od, ['original_lat', 'original_log'], _cs, ['lat', 'lng'])
    arr_dest_dis_to_cs = pairwise_distance(_od, ['destination_lat', 'destination_log'], _cs, ['lat', 'lng'])
    # Charging station capacity preparation.
    capacity = np.repeat(_cs['chg_points'].values.reshape(1, -1), _od.shape[0], axis=0)

    # Feature 1: distance of transition
    od_dis = vec_hs_dis.haversine_np(_od['original_log'], _od['original_lat'],
                                     _od['destination_log'], _od['destination_lat'])
    # Feature 2: distances of n nearest charging stations for transition origin.
    arr_origin_dis_to_cs_feature = arr_origin_dis_to_cs
    arr_origin_dis_to_cs_feature.sort(axis=1)
    # Feature 3: capacity of n nearest charging stations for transition origin.
    arr_capacity_around_origin = np.take_along_axis(capacity, arr_origin_dis_to_cs.argsort(axis=1), axis=1)
    # Feature 4: distances of n nearest charging stations for transition destination.
    arr_dest_dis_to_cs_feature = arr_dest_dis_to_cs
    arr_dest_dis_to_cs_feature.sort(axis=1)
    # Feature 5: capacity of n nearest charging stations for transition origin.
    arr_capacity_around_dest = np.take_along_axis(capacity, arr_dest_dis_to_cs.argsort(axis=1), axis=1)
    return np.concatenate((_od["duration"].to_numpy().reshape(-1, 1), od_dis.to_numpy().reshape(-1, 1),
                           arr_origin_dis_to_cs_feature[:, :_neighbor_num],
                           arr_capacity_around_origin[:, :_neighbor_num],
                           arr_dest_dis_to_cs_feature[:, :_neighbor_num],
                           arr_capacity_around_dest[:, :_neighbor_num]), axis=1)


if __name__ == '__main__':
    display.configure_logging()
    display.configure_pandas()

    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    # Hyper-parameters
    m, n = 100, 200
    neighbor_num = 3

    # ################### p2d train ########################
    # Load all transactions
    df_train_od = data_loader.load_od(scale='full', common=False)
    # Load ET transactions
    df_train_et_od = data_loader.load_od(scale='full', common=True)
    # Load CS (charging station) information.
    df_train_cs, _ = data_loader.load_cs(scale='part', date=datetime(2014, 7, 1))
    df_train_cs = df_train_cs.loc[~df_train_cs['cs_name'].isin(['LJDL', 'E04', 'BN0002', 'F11', 'S1', 'S2',
                                                                'F12', 'F13', 'F15'])]
    df_train_cs.reset_index(drop=True, inplace=True)

    # statistically count the transition demand.
    train_demand = extract_transition_demand(df_train_od)
    train_et_demand = extract_transition_demand(df_train_et_od)
    # merge the demand
    df_train_demands = build_composite_demands(train_demand, train_et_demand)
    # attach to the transactions
    df_train_unique_od = build_final_transitions(df_train_od, df_train_demands)
    # Build feature
    p2d_train_feature = feature_extraction(df_train_unique_od, df_train_cs)

    # ################### p2d val ########################
    val_od = pd.read_csv("data/od/201706_od.csv")
    val_et_od = pd.read_csv("data/od/201706_et_od.csv")
    df_val_cs = pd.read_csv("data/charging_station/ChargeLocation201706_wgs84.csv")

    # Feature extraction
    train_fuel_unique_od = build_final_transitions(df_train_od, train_demand)
    p2d_val_feature = feature_extraction(train_fuel_unique_od, df_val_cs, _neighbor_num=neighbor_num)

    # Demand ground truth
    val_et_demand = extract_transition_demand(val_et_od, threshold=0)
    p2d_val_gt = pd.merge(train_demand.rename(columns={"demand": "demand_all"}),
                          val_et_demand.rename(columns={"demand": "demand_17_et"}),
                          left_on=['original_cube', 'destination_cube'], right_on=['original_cube', 'destination_cube'],
                          how="left").fillna(0)
    et_14_demand = extract_transition_demand(df_train_et_od, threshold=0)
    p2d_val_gt = pd.merge(p2d_val_gt, et_14_demand.rename(columns={"demand": "demand_14_et"}),
                          on=['original_cube', 'destination_cube'], how="left").fillna(0)

    # ################### Save to local ########################
    np.save(conf["mobility"]["transition"]["utility_xgboost"]["p2d_train_feature"], p2d_train_feature)
    df_train_demands.to_csv(conf["mobility"]["transition"]["utility_xgboost"]["p2d_train_gt"], index=False)
    train_fuel_unique_od.to_csv(conf["mobility"]["transition"]["utility_xgboost"]["p2d_val_od"], index=False)
    np.save(conf["mobility"]["transition"]["utility_xgboost"]["p2d_val_feature"], p2d_val_feature)
    p2d_val_gt.to_csv(conf["mobility"]["transition"]["utility_xgboost"]["p2d_val_gt"], index=False)
