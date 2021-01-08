import pandas as pd


def search_ranges(range_record, trajectories, range_id='Licence', trajectory_id='plate',
                  begin_time='begin_time', end_time='end_time'):
    _id = range_record[range_id]
    _s_t, _e_t = range_record[begin_time], range_record[end_time]
    _id_s = trajectories[trajectory_id].searchsorted(_id, side='left')
    _id_e = trajectories[trajectory_id].searchsorted(_id, side='right')
    _temp = trajectories['timestamp'].iloc[_id_s: _id_e]
    _s_i = _id_s + _temp.searchsorted(_s_t, side='left')
    _e_i = _id_s + _temp.searchsorted(_e_t, side='right')
    return _s_i, _e_i
