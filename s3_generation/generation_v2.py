import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def single_period_generation():
    pass


if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    cs = pd.read_csv(conf['cs']['val'], usecols=['lng', 'lat', 'chg_points'])
    w = {idx: np.zeros(v) for idx, v in enumerate(cs['chg_points'].to_list())}
