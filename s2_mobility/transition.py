import os
import pickle
import yaml
from tqdm import tqdm
import datetime
import utils.data_loader as data_loader


def count_transits(df_od, a, b, a_t, b_t):
    """
    Statistically compute the transit probability and mean duration.
    :param df_od: original od (transaction) DataFrame
    :param a: transition origin
    :param b: transition destination
    :param a_t:
    :param b_t:
    :return:
    """
    _transits_prob = []
    _transits_mean_time = []
    start_time = datetime.datetime(1, 1, 1, 0, 0, 0)
    duration = datetime.timedelta(minutes=20)

    # Calculate time in the trip
    df_od['time_duration'] = (df_od[b_t] - df_od[a_t]).dt.total_seconds()
    # Extract time of day
    df_od['time_of_day'] = df_od[a_t].dt.time
    df_od.sort_values('time_of_day', axis=0, inplace=True)

    for interval in range(int(datetime.timedelta(days=1) / duration)):
        begin_time = start_time + interval * duration
        end_time = begin_time + duration - datetime.timedelta(seconds=1)
        begin_time, end_time = begin_time.time(), end_time.time()
        begin_index = df_od['time_of_day'].searchsorted(begin_time, side='left')
        end_index = df_od['time_of_day'].searchsorted(end_time, side='right')

        interval_od = df_od.iloc[begin_index: end_index]

        # Input: interval_od, Output: transit, transit_tim e
        # transit_amount has MultiIndex, levels are load_label and drop_label, and value is records count
        transit_amount = interval_od.groupby([a, b]).size()
        # Use unstack transform MultiIndex count to transit matrix, shape of (load_hotspots, drop_hotspots)
        transit = transit_amount.unstack(level=-1)
        # Delete noise related data, which represented by -1
        transit.drop(-1, axis=0, inplace=True)
        transit.drop(-1, axis=1, inplace=True)

        time_using = interval_od.groupby([a, b])['time_duration'].mean()
        # Use unstack transform MultiIndex count to transit time matrix, shape of (load_hotspots, drop_hotspots)
        transit_time = time_using.unstack(level=-1)
        # Delete noise related data, which represented by -1
        transit_time.drop(-1, axis=0, inplace=True)
        transit_time.drop(-1, axis=1, inplace=True)

        transit_time = transit_time.loc[~transit.isna().all(axis=1)]
        transit = transit.loc[~transit.isna().all(axis=1)]

        _transits_prob.append(transit)
        _transits_mean_time.append(transit_time)

    return _transits_prob, _transits_mean_time


def p2d():
    ##################################################################################################################
    # Statistically count transit from pick-up to drop-off
    od_with_hs = data_loader.load_od(with_hotpots=True, version='v4')
    od_with_hs = od_with_hs.loc[:, ['Licence', 'begin_time', 'end_time', 'load_label', 'drop_label']].copy()

    _transits_prob, _transits_mean_time = count_transits(od_with_hs, 'load_label', 'drop_label', 'begin_time',
                                                         'end_time')
    return _transits_prob, _transits_mean_time


def d2p():
    ##################################################################################################################
    # Statistically count transit from last drop-off to pick-up
    # 重新加载od_with_hs数据, load charging events, load rest events
    od_with_hs = data_loader.load_od(with_hotpots=True, version='v4')
    od_with_hs = od_with_hs.loc[:, ['Licence', 'begin_time', 'end_time', 'load_label', 'drop_label']].copy()

    ce = data_loader.load_ce(version='v5_30min')
    common = data_loader.load_trajectory_od_intersection()
    ce = ce.loc[ce['licence'].isin(common)].reset_index(drop=True)

    rest_events = data_loader.load_rest()

    # Mark transactions' last drop-off
    od_with_hs['last_drop_time'] = od_with_hs['begin_time'].shift()
    od_with_hs['last_drop_label'] = od_with_hs['drop_label'].shift()
    od_with_hs.loc[od_with_hs['Licence'] != od_with_hs['Licence'].shift(), 'last_drop_time'] = None
    od_with_hs.loc[od_with_hs['Licence'] != od_with_hs['Licence'].shift(), 'last_drop_label'] = None

    # First mark whether it is after a charging event
    def add_whether_after_ce(license_ce, od_all=od_with_hs):
        begin_index = od_all['Licence'].searchsorted(license_ce.name, side='left')
        end_index = od_all['Licence'].searchsorted(license_ce.name, side='right')
        license_od = od_all.iloc[begin_index: end_index]
        od_count = len(license_od.index)
        for _, row in license_ce.iterrows():
            od_index = license_od['begin_time'].searchsorted(row['start_charging'])
            if od_index == od_count:
                break
            else:
                license_od.at[license_od.index[od_index], 'after_ce'] = True
        return license_od

    tqdm.pandas()
    od_with_hs['after_ce'] = False
    od_with_hs = ce.groupby('licence').progress_apply(add_whether_after_ce, od_all=od_with_hs).reset_index(drop=True)

    # Second mark whether it is after a rest event
    def add_whether_after_rest(license_od, all_rests=None):
        begin_index = all_rests['license'].searchsorted(license_od.name, side='left')
        end_index = all_rests['license'].searchsorted(license_od.name, side='right')
        license_rests = all_rests.iloc[begin_index: end_index]
        od_count = len(license_od.index)
        for _, row in license_rests.iterrows():
            od_index = license_od['begin_time'].searchsorted(row['start_time'])
            if od_index == od_count:
                break
            else:
                license_od.at[license_od.index[od_index], 'after_rest'] = True
        return license_od

    tqdm.pandas()
    od_with_hs['after_rest'] = False
    od_with_hs = od_with_hs.groupby('Licence').progress_apply(add_whether_after_rest, all_rests=rest_events)
    od_with_hs = od_with_hs.loc[~(od_with_hs['after_ce'] | od_with_hs['after_rest'])]
    _transits_prob, _transits_mean_time = count_transits(od_with_hs, 'last_drop_label', 'load_label', 'last_drop_time',
                                                         'begin_time')

    return _transits_prob, _transits_mean_time


if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])
    ##################################################################################################################
    # Statistically count transit from pick-up to drop-off
    p2d_transits_prob, p2d_transits_mean_time = p2d()

    # Save result to local
    # with open('../data/transit_matrix/p2d_v3.list_of_df', 'wb') as f:
    #     pickle.dump(transits, f)
    #
    # with open('../data/transit_matrix/p2d_time_v3.list_of_df', 'wb') as f:
    #     pickle.dump(transits_time, f)

    ##################################################################################################################
    # Statistically count transit from last drop-off to pick-up
    d2p_transits_prob, d2p_transits_mean_time = d2p()

    # Save result to local
    # with open('../data/transit_matrix/d2p_v3.list_of_df', 'wb') as f:
    #     pickle.dump(transits, f)
    #
    # with open('../data/transit_matrix/d2p_time_v3.list_of_df', 'wb') as f:
    #     pickle.dump(transits_time, f)
