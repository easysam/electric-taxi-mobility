import os
import yaml
import argparse
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import timedelta
from sklearn.preprocessing import normalize
from utils import display, od_utils
from s1_preprocessing.hotspot.hotpots_discovery_utils import generate_cube_index
from s2_mobility.transit_prediction.s1_data_preparation import make_d2p_od, extract_transition_demand
from s2_mobility.transit_prediction.s3_build_transition_pattern import matrix_completion


def select_window_od(_od, _windows_len, _window_idx, time_col='begin_time'):
    hour = (_window_idx * _windows_len) // timedelta(hours=1)
    s_min = ((_window_idx * _windows_len) % timedelta(hours=1)).seconds / 60
    e_min = ((_window_idx * _windows_len) % timedelta(hours=1) + _windows_len).seconds / 60
    _window_od = _od.loc[(_od[time_col].dt.hour == hour)
                         & (_od[time_col].dt.minute > s_min) & (_od[time_col].dt.minute < e_min)]
    return _window_od


if __name__ == '__main__':
    display.configure_pandas()
    # configure the working directory to the project root path
    with open("config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='p2d', choices=['p2d', 'd2p'])
    parser.add_argument('--window_num', type=int, default=24)
    args = parser.parse_args()
    window_len = timedelta(days=1) / args.window_num

    with open(conf['mobility']['transition']['idx_cube_100_200_inverse_map'], mode='rb') as f:
        idx_inverse_map = pickle.load(f)
    if 'p2d' == args.task:
        row_cubes, col_cubes = idx_inverse_map['p2d_p'], idx_inverse_map['p2d_d']
    else:
        row_cubes, col_cubes = idx_inverse_map['d2p_d'], idx_inverse_map['d2p_p']

    val_od = pd.read_parquet(conf['od']['raw1706_pqt'])
    val_od.rename(columns={'id': "Licence"}, inplace=True)
    if 'd2p' == args.task:
        val_od = make_d2p_od(val_od)
    val_od = od_utils.filter_in_bbox(val_od)
    val_od = generate_cube_index(val_od, m=100, n=200)
    transition_tensor = np.zeros((args.window_num, len(row_cubes), len(col_cubes)))
    for w in tqdm(range(args.window_num)):
        window_od = select_window_od(val_od, window_len, w, time_col='begin_time').copy()
        window_demand = extract_transition_demand(window_od, threshold=0)

        window_mat = window_demand.pivot(index='original_cube', columns='destination_cube', values='demand')
        window_mat.fillna(0, inplace=True)

        row = set(window_mat.index.tolist())
        col = set(window_mat.columns.astype(dtype=int).tolist())

        window_mat.drop(list(row - set(row_cubes.keys())), inplace=True)
        window_mat.drop(list(col - set(col_cubes.keys())), axis=1, inplace=True)
        window_mat.drop(window_mat.index[0 == window_mat.sum(axis=1)], inplace=True)

        missing_col = list(set(col_cubes.keys()) - col)
        window_mat[missing_col] = 0.
        window_mat = window_mat.reindex(sorted(window_mat.columns), axis=1)

        row = set(window_mat.index.tolist())
        missing_row = np.array(list(set(row_cubes.keys()) - row), dtype=np.float32)
        window_mat = matrix_completion(missing_row, window_mat)

        transition_tensor[w] = normalize(window_mat.to_numpy(), norm='l1')

    np.savez_compressed(conf['mobility']['transition'][args.task]['gt_transition_tensor'], transition_tensor)
