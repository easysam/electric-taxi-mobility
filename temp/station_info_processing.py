import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from utils import display

if __name__ == '__main__':
    display.configure_pandas()

    # configure the working directory to the project root path
    with open("../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    cs_wg = pd.read_csv('data/charging_station/ChargeLocation201706', header=None)
    cs_new = pd.read_csv(conf['cs']['val'])

    print(cs_wg.head(3))
    print(cs_new.head(3))

    distance = haversine_distances(np.radians(cs_wg[[2, 3]]), np.radians(cs_new[['lng', 'lat']])) * 6371.0088
    merged = pd.concat([cs_wg,cs_new.iloc[np.argmin(distance, axis=1), [0, 3]].reset_index(drop=True)], axis=1)
    merged['dis'] = np.min(distance, axis=1)
    print(merged)
