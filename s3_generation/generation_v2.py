import os
import yaml
import pickle
import logging
import argparse
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from datetime import datetime, timedelta
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import haversine_distances

from utils import display
from s1_preprocessing.hotspot.hotpots_discovery_utils import cube_to_coordinate
from s2_mobility.charging_behavior.where2charge import Where2Charge


def build_rest_schedule(_n=10000, _pattern=None, _days=7):
    """
    output: a list _schedule with _n length for _n instances.
    Each element in _schedule is another list, the resting time schedule for the instance.
    Time in the schedule if the second offset from 00: 00: 00 in the first day.
    """
    logging.info("Building rest schedule...")
    _times = _pattern['times'].iloc[_pattern['times'].index <= 3].reset_index()
    _raw = _pattern['distribution']
    _len = len(_raw)
    _frac = [_len // (_i + 1) for _i in range(3)]
    _d = [pd.Series(_raw).reset_index(),
          pd.Series([_v + _raw[(_k + _frac[1]) % _len] for _k, _v in enumerate(_raw)]).reset_index(),
          pd.Series([_v + _raw[(_k + _frac[2]) % _len] + _raw[(_k + _frac[2]) * 2 % _len] for _k, _v in
                     enumerate(_raw)]).reset_index()]

    _schedule = [[] for _ in range(_n)]
    times_schedule = _times['index'].sample(_n * _days, weights=_times[0], replace=True).to_numpy().reshape(_n, _days)
    _t = []
    for _i in range(3):
        _t.append((_d[_i]['index'].sample(_n * _days, weights=_d[_i][0], replace=True) % _frac[_i]).to_numpy())
    _t_i = [0] * 3
    for _i in range(_n):
        for _day in range(_days):
            _time = times_schedule[_i, _day]
            for _j in range(_time):
                _schedule[_i].append(_days * 24 * 60 + _t[_time - 1][_t_i[_time - 1]] + _j * _frac[_time - 1])
            _t_i[_time - 1] += 1
        _schedule[_i] = [_x * 60 for _x in _schedule[_i]]
    return _schedule


def single_period_generation(_id, ts, _loc, _r, traveled=0):
    state = 'empty'
    _trajectory = pd.DataFrame(columns=['event', 'timestamp', 'location', 'traveled', 'info1', 'info2'])
    while True:
        if 'empty' == state:
            # Determine whether to charge
            _lng, _lat = cube_to_coordinate(idx_map['d2p_d'][_loc])
            ts_datetime = init_t + timedelta(seconds=ts)
            time_of_day = ts_datetime.hour + ts_datetime.minute / 60 + ts_datetime.second / 3600
            dis_to_cs = haversine_distances(np.radians([[_lat, _lng]]), np.radians(cs_loc)) * EARTH_RADIUS
            whether2charge_f = [time_of_day, dis_to_cs.min(), dis_to_cs.max(), dis_to_cs.mean(), np.median(dis_to_cs),
                                traveled]
            whether2charge_f_scaled = whether2charge_scaler.transform(np.reshape(whether2charge_f, (1, -1)))
            to_charge = whether2charge.predict(whether2charge_f_scaled)
            if (to_charge & (traveled < 50)) | (~to_charge & (traveled > 200)):
                to_charge = ~to_charge
            if to_charge:
                where2charge_f = pd.DataFrame(index=range(len(cs)))
                where2charge_f['max_dis'] = dis_to_cs.max()
                where2charge_f['mean_dis'] = dis_to_cs.mean()
                where2charge_f['mid_dis'] = np.median(dis_to_cs)
                where2charge_f['min_dis'] = dis_to_cs.min()
                where2charge_f['traveled'] = traveled
                where2charge_f['distance'] = dis_to_cs.reshape((-1,))
                where2charge_f['weekday'] = 1 if ts_datetime.weekday() < 5 else 0
                where2charge_f['time_of_day'] = time_of_day
                where2charge_f['chg_points'] = cs['chg_points']
                data = torch.from_numpy(where2charge_f.to_numpy()).to(device).float()
                data = data.view(-1, len(cs.index), len(where2charge_f.columns))
                output = where2charge(data)
                output = softmax(output).view(-1).cpu().detach().numpy()
                station_idx = np.random.choice(len(output), 1, p=output).item()
                state = 'charging'
                continue
            # Determine whether to rest
            if ts > resting_schedule[_id][_r]:
                state = 'resting'
                continue
            # Move on to the occupied stated
            _loc_prev = _loc
            _loc = np.random.choice(np.arange(len(idx_map['d2p_p'])), size=1, p=d2p_prob[_loc_prev])
            ts += d2p_dur[_loc_prev][_loc]
            traveled += d2p_dis[_loc_prev][_loc]
            _trajectory.loc[len(_trajectory)] = ['pick-up', ts_datetime, idx_map['d2p_d'][_loc], traveled]
            state = 'occupied'
            break
        elif 'occupied' == state:
            ts_datetime = init_t + timedelta(seconds=ts)
            _loc_prev = _loc
            _loc = np.random.choice(np.arange(len(idx_map['p2d_d'])), size=1, p=p2d_prob[_loc_prev])
            ts += p2d_dur[_loc_prev][_loc]
            traveled += d2p_dis[_loc_prev][_loc]
            _trajectory.loc[len(_trajectory)] = ['pick-up', ts_datetime, idx_map['d2p_d'][_loc], traveled]
            state = 'empty'
            break
        elif 'resting' == state:
            ts_datetime = init_t + timedelta(seconds=ts)
            ts += rest_pattern['duration'][ts_datetime.hour]
            _r += 1
            _trajectory.loc[len(_trajectory)] = ['resting', ts_datetime, idx_map['d2p_d'][_loc], traveled,
                                                 rest_pattern['duration'][ts_datetime.hour]]
            state = 'empty'
            break
        elif 'charging' == state:
            _c = 120 * 60 * traveled / (250 * 1000)
            return _trajectory, ts, _loc, station_idx, _c, _r


if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])
    display.configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_schedule', action='store_true')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    EARTH_RADIUS = 6371.0088
    # Key parameter: ET number and charger distribution
    n = 10000
    cs = pd.read_csv(conf['cs']['val'], usecols=['lng', 'lat', 'chg_points'])
    cs_loc = cs[['lat', 'lng']].to_numpy()
    # Mobility pattern
    # 1. transition pattern
    p2d_prob = normalize(np.load(conf['mobility']['transition']['utility_xgboost']['p2d']['prob_mat']), norm='l1')
    p2d_dis = np.load(conf['mobility']['transition']['p2d']['distance'])
    p2d_dur = np.load(conf['mobility']['transition']['p2d']['duration'])
    d2p_prob = normalize(np.load(conf['mobility']['transition']['utility_xgboost']['d2p']['prob_mat']), norm='l1')
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
    where2charge.eval()
    softmax = torch.nn.Softmax(dim=1)
    # 3. resting pattern
    with open(conf['mobility']['resting'], 'rb') as f:
        # rest pattern: 1, times; 2, distribution; 3, duration.
        rest_pattern = pickle.load(f)

    print('p2d_p len: {}, head(1): {}'.format(len(idx_map['p2d_p']), idx_map['p2d_p'][1]))
    print('p2d_d len: {}, head(1): {}'.format(len(idx_map['p2d_d']), idx_map['p2d_d'][1]))
    print('d2p_d len: {}, head(1): {}'.format(len(idx_map['d2p_d']), idx_map['d2p_d'][1]))
    print('d2p_p len: {}, head(1): {}'.format(len(idx_map['d2p_p']), idx_map['d2p_p'][1]))
    print('d2p shape: {}'.format(d2p_prob.shape))
    print('p2d shape: {}'.format(p2d_prob.shape))
    # Initial variable
    init_t = datetime(2017, 6, 1, 0, 0, 0)
    init_l = np.random.randint(0, high=len(idx_map['d2p_d']), size=(n,))
    if args.local_schedule:
        with open(conf['generation']['schedule'], 'rb') as f:
            resting_schedule = pickle.load(f)
    else:
        resting_schedule = build_rest_schedule(n, rest_pattern)
        with open(conf['generation']['schedule'], 'wb') as f:
            pickle.dump(resting_schedule, f)
    w = {idx: np.zeros(v) for idx, v in enumerate(cs['chg_points'].to_list())}
    t = [0] * n
    r = [0] * n
    for i in tqdm(range(n)):
        single_period_generation(i, 0, init_l[i], 0)
