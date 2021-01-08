import os
import yaml
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import data_loader, display

if __name__ == '__main__':
    tqdm.pandas()
    display.configure_pandas()
    display.configure_logging()
    # configure the working directory to the project root path
    with open("../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    rest_events = pd.read_parquet('data/rest/rest_events.parquet')

    # 统计每天休息事件发生的次数
    plate_day_counts = rest_events.groupby('license').apply(lambda x: x.groupby('day').size())
    plate_day_counts = plate_day_counts.unstack(level=-1).fillna(0)
    plate_times = plate_day_counts.mean(axis=1).round().astype(int)
    resting_times = (plate_times.value_counts() / plate_times.size).sort_index()

    # Fig 1: 考察开始休息时间的分布
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # Data preparation
    rest_events['time_of_day'] = rest_events['start_time'].dt.hour + rest_events['start_time'].dt.minute / 60
    ce = data_loader.load_ce(version='v5_30min')
    common = data_loader.load_trajectory_od_intersection()
    ce = ce.loc[ce['licence'].isin(common)].reset_index(drop=True)
    ce['time_of_day'] = ce['start_charging'].dt.hour + ce['start_charging'].dt.minute / 60
    # Plot
    ce.plot(y='time_of_day', kind='hist', bins=24 * 12, ax=ax, secondary_y=False, histtype='step', legend=True,
            density=True, label='Charging events')
    rest_events['time_of_day'].plot(kind='hist', bins=24 * 12, ax=ax, density=True, legend=True,
                                    grid=False, histtype='step', label='Resting events')
    ax.set_xticks(range(0, 25, 2))
    ax.margins(x=0)
    ax.set_title('Distribution of rest events and charging events')
    ax.set_xlabel('Time of day/ hour')
    ax.set_ylabel('Probability', )
    plt.show()
    # Resting event start time distribution
    count, division = np.histogram(rest_events['time_of_day'], bins=24 * 60)
    print(count)

    # Figure 2: 考察休息时长的变化
    # Data preparation
    rest_events['rest_length'] = rest_events['duration'] / 3600
    rest_events['hour'] = rest_events['start_time'].dt.hour
    rest_events['interval_in_hour'] = rest_events['start_time'].dt.minute // 30
    rest_dur = rest_events.groupby(['hour', ])['rest_length'].mean()
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    rest_dur.plot(style='.-', ax=ax, grid=True)
    ax.set_xticks(range(0, 24, 2))
    ax.margins(x=0)
    ax.set_title('Rest length in time of day')
    ax.set_xlabel('time of day/hour')
    ax.set_ylabel('rest length')
    plt.show()
    print(rest_events.groupby(['hour', ])['rest_length'].mean())

    with open(conf['mobility']['resting'], 'wb') as f:
        pickle.dump({'times': resting_times, 'distribution': count, 'duration': rest_dur}, f)
