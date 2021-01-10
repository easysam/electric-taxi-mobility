import os
import yaml
import pickle
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from s2_mobility.charging_behavior.where2charge import Where2Charge
from datetime import datetime, timedelta


def build_rest_schedule(_n=10000, _pattern=None, _days=7):
    """
    output: a list _schedule with _n length for _n instances.
    Each element in _schedule is another list, the resting time schedule for the instance.
    Time in the schedule if the second offset from 00: 00: 00 in the first day.
    """
    _times = _pattern['times'].iloc[_pattern['times'].index <= 3].reset_index()
    _raw = _pattern['distribution']
    _len = len(_raw)
    _frac = [_len // (_i + 1) for _i in range(3)]
    _d = [pd.Series(_raw).reset_index()]
    _d.append(pd.Series([_v + _raw[(_k + _frac[1]) % _len] for _k, _v in enumerate(_raw)]).reset_index())
    _d.append(pd.Series([_v + _raw[(_k + _frac[2]) % _len] + _raw[(_k + _frac[2]) * 2 % _len] for _k, _v in
                         enumerate(_raw)]).reset_index())
    _dur = _pattern['duration']

    _schedule = [[]] * _n
    times_schedule = _times['index'].sample(_n * _days, weights=_times[0], replace=True).to_numpy().reshape(_n, _days)
    _t = []
    for _i in range(3):
        _t.append((_d[_i]['index'].sample(_n * _days, weights=_d[_i][0], replace=True) % (_len // (_i + 1))).to_numpy())
    _t_i = [0] * 3
    for _i in range(_n):
        for _day in range(_days):
            _time = times_schedule[_i, _day]
            for _j in range(_time):
                _schedule[_i].append(_days * 24 * 60 + _t[_time - 1][_t_i[_time - 1]] + _j * _frac[_time - 1])
            _t_i[_time - 1] += 1
        _schedule[_i] = [_x * 60 for _x in _schedule[_i]]
    return _schedule


def single_period_generation(_t, _loc, _r, traveled=0):
    return None, None, None, None, None, None


if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Key parameter: ET number and charger distribution
    n = 10000
    cs = pd.read_csv(conf['cs']['val'], usecols=['lng', 'lat', 'chg_points'])
    # Mobility pattern
    # 1. transition pattern
    p2d_prob = np.load(conf['mobility']['transition']['utility_xgboost']['p2d']['prob_mat'])
    p2d_dis = np.load(conf['mobility']['transition']['p2d']['distance'])
    p2d_dur = np.load(conf['mobility']['transition']['p2d']['duration'])
    d2p_prob = np.load(conf['mobility']['transition']['utility_xgboost']['d2p']['prob_mat'])
    d2p_dis = np.load(conf['mobility']['transition']['d2p']['distance'])
    d2p_dur = np.load(conf['mobility']['transition']['d2p']['duration'])
    with open(conf['mobility']['transition']['idx_cube_100_200_map'], mode='rb') as f:
        idx_map = pickle.load(f)
    # 2. charging pattern
    whether2charge = xgb.XGBClassifier(verbosity=1, max_depth=10, learning_rate=0.01, n_estimators=500,
                                       scale_pos_weight=10)
    whether2charge.load_model(conf['mobility']['charge']['whether_xgb'])
    with open(conf['mobility']['charge']['whether_xgb_scaler'], 'rb') as f:
        whether2charge_scaler = pickle.load(f)
    where2charge = Where2Charge()
    where2charge.to(device)
    where2charge.load_state_dict(torch.load(conf["mobility"]["charge"]["where_model"] + '.best'))
    # 3. resting pattern
    with open(conf['mobility']['resting'], 'rb') as f:
        # rest pattern: 1, times; 2, distribution; 3, duration.
        rest_pattern = pickle.load(f)

    # Initial variable
    init_l = np.random.randint(0, high=len(idx_map['d2p_d']), size=(n,))
    resting_schedule = build_rest_schedule(n, rest_pattern)
    w = {idx: np.zeros(v) for idx, v in enumerate(cs['chg_points'].to_list())}
    t = [0] * n
    r = [0] * n
    for i in range(n):
        single_period_generation(0, init_l[i], 0)
