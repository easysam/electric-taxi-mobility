import os
import yaml
import pickle
import logging
import argparse
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from tqdm import tqdm
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
                _schedule[_i].append(_day * 24 * 60 + _t[_time - 1][_t_i[_time - 1]] + _j * _frac[_time - 1])
            _t_i[_time - 1] += 1
        _schedule[_i] = [_x * 60 for _x in _schedule[_i]]
    return _schedule


def cube_convert(_id, _from, _to):
    # idx_map idx_inv_map
    _temp = idx_map[_from][_id]
    return idx_inv_map[_to][_temp]


def single_period_generation(_id, ts, _loc, _r, _traveled=0):
    state = 'empty'
    _trajectory = pd.DataFrame(columns=['event', 'timestamp', 'location', 'traveled', 'station', 'queuing', 'charging'])
    while True:
        if 'empty' == state:
            # Determine whether to charge
            _lng, _lat = cube_to_coordinate(idx_map['d2p_d'][_loc], to_geodetic=True, m=100, n=200)
            ts_datetime = init_t + timedelta(seconds=ts)
            time_of_day = ts_datetime.hour + ts_datetime.minute / 60 + ts_datetime.second / 3600
            dis_to_cs = haversine_distances(np.radians([[_lat, _lng]]), np.radians(cs_loc)) * EARTH_RADIUS
            whether2charge_f = [time_of_day, dis_to_cs.min(), dis_to_cs.max(), dis_to_cs.mean(), np.median(dis_to_cs),
                                _traveled / 1000]
            whether2charge_f_scaled = whether2charge_scaler.transform(np.reshape(whether2charge_f, (1, -1)))
            to_charge = whether2charge.predict(whether2charge_f_scaled).item()
            if (to_charge & (_traveled / 1000 < 150)) | (~to_charge & (_traveled / 1000 > 200)):
                to_charge = 1 - to_charge
            if to_charge:
                where2charge_f = pd.DataFrame(index=range(len(cs)))
                where2charge_f['max_dis'] = dis_to_cs.max()
                where2charge_f['mean_dis'] = dis_to_cs.mean()
                where2charge_f['mid_dis'] = np.median(dis_to_cs)
                where2charge_f['min_dis'] = dis_to_cs.min()
                where2charge_f['traveled'] = _traveled / 1000
                where2charge_f['distance'] = dis_to_cs.reshape((-1,))
                where2charge_f['weekday'] = 1 if ts_datetime.weekday() < 5 else 0
                where2charge_f['time_of_day'] = time_of_day
                where2charge_f['chg_points'] = cs['chg_points']
                data = torch.from_numpy(where2charge_f.to_numpy()).to(device).float()
                data = data.view(-1, len(cs.index), len(where2charge_f.columns))
                output = where2charge(data)
                # output = softmax(output).view(-1).cpu().detach().numpy()
                output = output.view(-1).cpu().detach().numpy()
                output = normalize(output.reshape(1, -1), norm='l1').reshape(-1)
                station_idx = np.random.choice(len(output), 1, p=output).item()
                state = 'charging'
                continue
            # Determine whether to rest
            if len(resting_schedule[_id]) < _r:
                if ts > resting_schedule[_id][_r]:
                    state = 'resting'
                continue
            # Move on to the occupied stated
            _loc_prev = _loc
            _loc = np.random.choice(np.arange(len(idx_map['d2p_p'])), size=1, p=d2p_prob[_loc_prev]).item()
            ts += 10 * (d2p_dur[_loc_prev][_loc] if d2p_dur[_loc_prev][_loc] != 0 else 20)
            _traveled += 2 * (d2p_dis[_loc_prev][_loc] if d2p_dis[_loc_prev][_loc] != 0 else 250)
            _trajectory.loc[len(_trajectory)] = ['pick-up', ts_datetime, idx_map['d2p_p'][_loc], _traveled, None, None,
                                                 None]
            state = 'occupied'
            _loc = cube_convert(_loc, 'd2p_p', 'p2d_p')
            continue
        elif 'occupied' == state:
            ts_datetime = init_t + timedelta(seconds=ts)
            _loc_prev = _loc
            _loc = np.random.choice(np.arange(len(idx_map['p2d_d'])), size=1, p=p2d_prob[_loc_prev]).item()
            ts += p2d_dur[_loc_prev][_loc] if p2d_dur[_loc_prev][_loc] != 0 else 20
            _traveled += d2p_dis[_loc_prev][_loc] if d2p_dis[_loc_prev][_loc] != 0 else 250
            _trajectory.loc[len(_trajectory)] = ['drop-off', ts_datetime, idx_map['p2d_d'][_loc], _traveled, None, None,
                                                 None]
            state = 'empty'
            _loc = cube_convert(_loc, 'p2d_d', 'd2p_d')
            continue
        elif 'resting' == state:
            ts_datetime = init_t + timedelta(seconds=ts)
            ts += rest_pattern['duration'][ts_datetime.hour] if rest_pattern['duration'][ts_datetime.hour] != 0 else 20
            _r += 1
            _trajectory.loc[len(_trajectory)] = ['resting', ts_datetime, idx_map['d2p_d'][_loc], _traveled,
                                                 rest_pattern['duration'][ts_datetime.hour], None, None]
            state = 'empty'
            continue
        elif 'charging' == state:
            _c = 120 * 60 * _traveled / (250 * 1000)
            return _trajectory, ts, _loc, station_idx, _traveled, _c, _r


if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])
    display.configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--day', type=int, default=7)
    parser.add_argument('--local_schedule', action='store_true')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    EARTH_RADIUS = 6371.0088
    # Key parameter: ET number and charger distribution
    n = args.n
    day = args.day
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
    with open(conf['mobility']['transition']['idx_cube_100_200_inverse_map'], mode='rb') as f:
        idx_inv_map = pickle.load(f)
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
    resting_schedule = build_rest_schedule(n, rest_pattern, _days=day)
    w = {idx: np.zeros(v) for idx, v in enumerate(cs['chg_points'].to_list())}
    trajectories = [None for _ in range(n)]
    t = [None for _ in range(n)]
    r = [None for _ in range(n)]
    loc = [None for _ in range(n)]
    s = [None for _ in range(n)]
    c = [None for _ in range(n)]
    traveled = [None for _ in range(n)]
    for i in tqdm(range(n)):
        # _trajectory, ts, _loc, station_idx, _c, _r
        trajectories[i], t[i], loc[i], s[i], traveled[i], c[i], r[i] = single_period_generation(i, 0, init_l[i], 0)
    t_previous = 0
    while True:
        i = np.argmin(t).item()
        if t[i] > day * 24 * 60 * 60:
            break
        else:
            print("\rGeneration progress: {:.1f}%".format(t[i] / (day * 24 * 60 * 60) * 100), end='')
        for station in w:
            w[station] = np.max((w[station] - (t[i] - t_previous)).reshape(-1, 1), axis=-1, initial=0).reshape(-1)
        t_previous = t[i]
        k = np.argmin(w[s[i]]).item()
        q = w[s[i]][k]
        w[s[i]][k] += c[i]
        trajectories[i].loc[len(trajectories[i])] = ['charging', init_t + timedelta(seconds=t[i]),
                                                     idx_map['d2p_d'][loc[i]], traveled[i], s[i], q, c[i]]
        sub_trajectories, t[i], loc[i], s[i], traveled[i], c[i], r[i] = single_period_generation(i, t[i] + q + c[i],
                                                                                                 init_l[i], r[i])
        trajectories[i] = pd.concat([trajectories[i], sub_trajectories])
    pd.concat(trajectories, keys=np.arange(n), names=['id', 'foo']).droplevel('foo').to_parquet(
        conf['generation']['result'])
