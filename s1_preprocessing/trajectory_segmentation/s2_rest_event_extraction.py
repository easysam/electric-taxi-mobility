import os
import yaml
import pandas as pd
from tqdm import tqdm
from utils import display

if __name__ == '__main__':
    tqdm.pandas()
    display.configure_pandas()
    display.configure_logging()
    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    # Load data
    trajectories = pd.read_parquet('data/trajectory/statuses')
    # print("If ce begin_time conflict with od?", (trajectories['charging'] & trajectories['occupied']).any())
    # print("If ce start_charging conflict with od?", (trajectories['actual_charging'] & trajectories['occupied']).any())
    # print("If ce arrival_time conflict with od?", (trajectories['queuing'] & trajectories['occupied']).any())
    print(trajectories.head(3))

    # Make a special trajectory that switch timestamp to last one if point is end of a big interval.
    trajectories['last_timestamp'] = trajectories['timestamp'].shift()
    trajectories.loc[trajectories['plate'] != trajectories['plate'].shift(), 'last_timestamp'] = None
    trajectories['begin_time'] = trajectories['timestamp']
    trajectories.loc[trajectories['big_dur'] & trajectories['valid'], 'begin_time'] = \
        trajectories.loc[trajectories['big_dur'] & trajectories['valid'], 'last_timestamp']
    # Group to stay trajectories
    trajectories['grp'] = ((trajectories['stop'] != trajectories['stop'].shift())
                           | (trajectories['plate'] != trajectories['plate'].shift())
                           | (trajectories['occupied'] != trajectories['occupied'].shift())
                           | (trajectories['charging'] != trajectories['charging'].shift())).cumsum()
    trajectories['day'] = trajectories['timestamp'].dt.day
    stay_trajectory = pd.DataFrame({
        'license': trajectories.groupby('grp')['plate'].first(),
        'start_time': trajectories.groupby('grp')['begin_time'].first(),
        'end_time': trajectories.groupby('grp')['timestamp'].last(),
        'duration': (trajectories.groupby('grp')['timestamp'].last() -
                     trajectories.groupby('grp')['begin_time'].first()).dt.total_seconds(),
        'Longitude': trajectories.groupby('grp')['longitude'].mean(),
        'Latitude': trajectories.groupby('grp')['latitude'].mean(),
        'occupied': trajectories.groupby('grp')['occupied'].first(),
        'stop': trajectories.groupby('grp')['stop'].first(),
        'charging': trajectories.groupby('grp')['charging'].first(),
        'day': trajectories.groupby('grp')['day'].first(),
        'have_big_dur': trajectories.groupby('grp')['big_dur'].any(),
    })
    rest_events = stay_trajectory.loc[
        ~stay_trajectory['occupied'] & stay_trajectory['stop'] & ~stay_trajectory['charging']
        & (stay_trajectory['duration'] > 1800)]

    rest_events.to_csv('data/rest/rest_events.csv', index=False)
    rest_events.to_parquet('data/rest/rest_events.parquet')
