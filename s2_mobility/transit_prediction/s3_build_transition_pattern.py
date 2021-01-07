import os
import yaml
import json
import argparse
import pickle
import urllib.request
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import haversine_distances

from utils import display
from s1_preprocessing.hotspot.hotpots_discovery_utils import cube_to_coordinate


def find_nearest_cube(targets, candidates):
    t_lng, t_lat = cube_to_coordinate(targets, to_geodetic=True, m=100, n=200)
    t_loc = np.concatenate((t_lat.reshape(-1, 1), t_lng.reshape(-1, 1)), axis=1)
    c_lng, c_lat = cube_to_coordinate(candidates, to_geodetic=True, m=100, n=200)
    c_loc = np.concatenate((c_lat.reshape(-1, 1), c_lng.reshape(-1, 1)), axis=1)
    nearest_idx = haversine_distances(np.radians(t_loc), np.radians(c_loc)).argmin(axis=1)
    return nearest_idx


def matrix_completion(missing_cube, raw_mat):
    missing_cube_nearest = find_nearest_cube(missing_cube, np.array(raw_mat.index.tolist()))
    supplement_cube = raw_mat.iloc[missing_cube_nearest]
    supplement_cube.set_index(missing_cube.astype(dtype=int), inplace=True)
    completion_mat = pd.concat([raw_mat, supplement_cube]).sort_index()
    return completion_mat


def fetch_cost(transition):
    _a, _b, _c, _d = transition[['o_lng', 'o_lat', 'd_lng', 'd_lat']]
    try:
        contents = urllib.request.urlopen(
            "http://router.project-osrm.org/route/v1/driving/{},{};{},{}?overview=false".format(_a, _b, _c, _d)
        ).read()
    except:
        print('e', end='')
        return pd.Series([None, None], index=['path_len', 'travel_time'])
    my_json = contents.decode('utf8').replace("'", '"')
    dis = json.loads(my_json)['routes'][0]['distance']
    duration = json.loads(my_json)['routes'][0]['duration']
    return pd.Series([dis, duration], index=['path_len', 'travel_time'])


def build_matrices(_orig_cube, _dest_cube, _transition_cost):
    _idx = pd.MultiIndex.from_arrays([_orig_cube, _dest_cube], names=('o_cube', 'd_cube'))
    _dis = _transition_cost.loc[_idx].reset_index().pivot(index='o_cube', columns='d_cube',
                                                          values='path_len').fillna(0)
    _duration = transition_cost.loc[_idx].reset_index().pivot(index='o_cube', columns='d_cube',
                                                              values='travel_time').fillna(0)
    return _dis, _duration


if __name__ == '__main__':
    tqdm.pandas()
    display.configure_logging()
    display.configure_pandas()
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()
    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    p2d_raw_prob_mat = pd.read_csv(conf['mobility']['transition']['utility_xgboost']['p2d']['prob_mat_incomplete'],
                                   index_col=0)
    d2p_raw_prob_mat = pd.read_csv(conf['mobility']['transition']['utility_xgboost']['d2p']['prob_mat_incomplete'],
                                   index_col=0)
    p_in_p2d = set(p2d_raw_prob_mat.index.tolist())
    d_in_p2d = set(p2d_raw_prob_mat.columns.astype(dtype=int).tolist())
    d_in_d2p = set(d2p_raw_prob_mat.index.tolist())
    p_in_d2p = set(d2p_raw_prob_mat.columns.astype(dtype=int).tolist())

    missing_p = np.array(list(p_in_d2p - p_in_p2d), dtype=np.float32)
    p2d_prob_mat = matrix_completion(missing_p, p2d_raw_prob_mat)
    missing_d = np.array(list(d_in_p2d - d_in_d2p), dtype=np.float32)
    d2p_prob_mat = matrix_completion(missing_d, d2p_raw_prob_mat)

    orig_idx_a, dest_idx_a = np.nonzero(p2d_prob_mat.to_numpy())
    orig_idx_b, dest_idx_b = np.nonzero(d2p_prob_mat.to_numpy())
    orig_cube_a, dest_cube_a = p2d_prob_mat.index[orig_idx_a], p2d_prob_mat.columns[dest_idx_a].astype(dtype=int)
    orig_cube_b, dest_cube_b = d2p_prob_mat.index[orig_idx_b], d2p_prob_mat.columns[dest_idx_b].astype(dtype=int)
    orig_cube = np.concatenate((orig_cube_a, orig_cube_b)).reshape(-1, 1)
    dest_cube = np.concatenate((dest_cube_a, dest_cube_b)).reshape(-1, 1)
    transition_cost = pd.DataFrame(np.concatenate((orig_cube, dest_cube), axis=1), columns=['o_cube', 'd_cube'])
    transition_cost.drop_duplicates(inplace=True, ignore_index=True)
    transition_cost['o_lng'], transition_cost['o_lat'] = cube_to_coordinate(transition_cost['o_cube'], to_geodetic=True,
                                                                            m=100, n=200)
    transition_cost['d_lng'], transition_cost['d_lat'] = cube_to_coordinate(transition_cost['d_cube'], to_geodetic=True,
                                                                            m=100, n=200)

    cache_path = conf['mobility']['transition']['transition_cost']['cube_100_200']
    if args.local:
        transition_cost = pd.read_csv(cache_path, index_col=0)
    else:
        ret = transition_cost.progress_apply(fetch_cost, axis=1)
        transition_cost = pd.concat([transition_cost, ret], axis=1)
        transition_cost.to_csv(cache_path, index=False)
    transition_cost.set_index(['o_cube', 'd_cube'], inplace=True)

    p2d_dis, p2d_duration = build_matrices(orig_cube_a, dest_cube_a, transition_cost)
    d2p_dis, d2p_duration = build_matrices(orig_cube_b, dest_cube_b, transition_cost)

    idx_map = {'p2d_p': {_k: _v for _k, _v in enumerate(p2d_prob_mat.index.tolist())},
               'p2d_d': {_k: _v for _k, _v in enumerate(p2d_prob_mat.columns.astype(dtype=int).tolist())},
               'd2p_d': {_k: _v for _k, _v in enumerate(d2p_prob_mat.index.tolist())},
               'd2p_p': {_k: _v for _k, _v in enumerate(d2p_prob_mat.columns.astype(dtype=int).tolist())}}

    with open(conf['mobility']['transition']['idx_cube_100_200_map'], mode='wb') as f:
        pickle.dump(idx_map, f)
    np.save(conf['mobility']['transition']['utility_xgboost']['p2d']['prob_mat'], p2d_prob_mat.to_numpy())
    np.save(conf['mobility']['transition']['p2d']['distance'], p2d_dis.to_numpy())
    np.save(conf['mobility']['transition']['p2d']['duration'], p2d_duration.to_numpy())

    np.save(conf['mobility']['transition']['utility_xgboost']['d2p']['prob_mat'], p2d_prob_mat.to_numpy())
    np.save(conf['mobility']['transition']['d2p']['distance'], d2p_dis.to_numpy())
    np.save(conf['mobility']['transition']['d2p']['duration'], d2p_duration.to_numpy())
