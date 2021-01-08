import os
import yaml
import pandas as pd
from utils import display

if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    # To convert trajectory csv file to parquet file
    # trajectories = pd.read_csv('data/history_trajectories.csv', dtype={'velocity': np.int32, 'status': np.int8})
    # trajectories.drop(columns=['color', 'status'], inplace=True)
    # trajectories['timestamp'] = pd.to_datetime(trajectories['timestamp'])
    # trajectories.to_parquet('data/trajectory/raw')

    # To convert od csv file to parquet file
    # od = pd.read_csv('data/transaction_201407.csv')
    # od['begin_time'] = pd.to_datetime(od['begin_time'])
    # od['end_time'] = pd.to_datetime(od['end_time'])
    # od.to_parquet('data/od/raw')
