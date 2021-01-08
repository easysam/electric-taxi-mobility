import os
import yaml
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def single_period_generation(traveled=0,):
    pass


if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    n = 10000
    cs = pd.read_csv(conf['cs']['val'], usecols=['lng', 'lat', 'chg_points'])
    w = {idx: np.zeros(v) for idx, v in enumerate(cs['chg_points'].to_list())}
    p2d_prob = np.load(conf['mobility']['transition']['utility_xgboost']['p2d']['prob_mat'])
    p2d_dis = np.load(conf['mobility']['transition']['p2d']['distance'])
    p2d_dur = np.load(conf['mobility']['transition']['p2d']['duration'])
    d2p_prob = np.load(conf['mobility']['transition']['utility_xgboost']['d2p']['prob_mat'])
    d2p_dis = np.load(conf['mobility']['transition']['d2p']['distance'])
    d2p_dur = np.load(conf['mobility']['transition']['d2p']['duration'])
    with open(conf['mobility']['transition']['idx_cube_100_200_map'], mode='rb') as f:
        idx_map = pickle.load(f)
    init_l = np.random.randint(0, high=len(idx_map['d2p_d']), size=(n,))
