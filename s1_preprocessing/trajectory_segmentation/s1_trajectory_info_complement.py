import os
import yaml
import pickle
import numpy as np
import pandas as pd
from utils import display, vector_haversine_distances as hav_dis, trajectory_util
from tqdm import tqdm

if __name__ == '__main__':
    tqdm.pandas()
    display.configure_pandas()
    display.configure_logging()
    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    trajectories = pd.read_parquet('data/trajectory/raw')
    od = pd.read_parquet('data/od/raw')
    with open('data/201407et_list.pickle', 'rb') as f:
        et14_list = pickle.load(f)
    et_od = od.loc[od['Licence'].isin(et14_list)]

    # 1. Add occupied flag
    ranges = et_od.iloc[:].progress_apply(trajectory_util.search_ranges, axis=1, args=(trajectories,),
                                          range_id='Licence', trajectory_id='plate', begin_time='begin_time',
                                          end_time='end_time')
    occupied_idx = np.concatenate([np.array(list(range(_a, _b))) for _a, _b in ranges]).astype(int)
    trajectories['occupied'] = False
    trajectories.iloc[occupied_idx, -1] = True
    # 2. Add dis2pre flag
    trajectories['dis2pre'] = hav_dis.haversine_np(trajectories['longitude'].shift(), trajectories['latitude'].shift(),
                                                   trajectories['longitude'], trajectories['latitude']
                                                   ).astype(np.float32)
    # 3. Add dur2pre flag
    trajectories['dur2pre'] = (trajectories['timestamp']
                               - trajectories['timestamp'].shift()).dt.total_seconds().astype(np.float32)
    trajectories.loc[trajectories['plate'] != trajectories['plate'].shift(), ['dis2pre', 'dur2pre']] = None
    # 4. Add big_dur flag
    trajectories['big_dur'] = trajectories['dur2pre'] > 1800
    # 5. Add valid flag
    trajectories['valid'] = ~(trajectories['big_dur'] & (trajectories['dis2pre'] > 0.5))
    # 6. Add stop flag
    trajectories['stop'] = (((trajectories['velocity'] == 0) & ~trajectories['big_dur']) |
                            (trajectories['big_dur'] & trajectories['valid']))
    # 7. Add charging flag
    ce = pd.read_csv('data/charging_event/ce_v5_30min.csv', parse_dates=['arrival_time', 'start_charging', 'end_time'],
                     usecols=['licence', 'arrival_time', 'begin_time', 'start_charging', 'end_time'])
    ranges = ce.progress_apply(trajectory_util.search_ranges, axis=1, args=(trajectories,),
                               range_id='licence', trajectory_id='plate', begin_time='begin_time', end_time='end_time')
    ce_idx = np.concatenate([np.array(list(range(_a, _b))) for _a, _b in ranges]).astype(int)
    trajectories['charging'] = False
    trajectories.iloc[ce_idx, -1] = True
    # 8. Add queuing flag
    ranges = ce.progress_apply(trajectory_util.search_ranges, axis=1, args=(trajectories,),
                               range_id='licence', trajectory_id='plate', begin_time='arrival_time',
                               end_time='start_charging')
    ce_idx = np.concatenate([np.array(list(range(_a, _b))) for _a, _b in ranges]).astype(int)
    trajectories['queuing'] = False
    trajectories.iloc[ce_idx, -1] = True
    # 9. Add actual_charging flag
    ranges = ce.progress_apply(trajectory_util.search_ranges, axis=1, args=(trajectories,),
                               range_id='licence', trajectory_id='plate', begin_time='start_charging',
                               end_time='end_time')
    ce_idx = np.concatenate([np.array(list(range(_a, _b))) for _a, _b in ranges]).astype(int)
    trajectories['actual_charging'] = False
    trajectories.iloc[ce_idx, -1] = True

    # Save to local
    trajectories.to_parquet('data/trajectory/statuses')
