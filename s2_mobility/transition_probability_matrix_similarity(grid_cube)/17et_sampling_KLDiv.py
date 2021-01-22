import os
import yaml
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, normalize
from utils import display, od_utils
from s1_preprocessing.hotspot.hotpots_discovery_utils import generate_cube_index
from s2_mobility.transit_prediction.s2_utility_XGBoost_train import kl

if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])
    display.configure_pandas()

    od = pd.read_parquet(conf['od']['raw1706_pqt'])
    od = od_utils.filter_in_bbox(od)
    od = generate_cube_index(od, m=100, n=200)
    od_pairs_demand = pd.read_csv(conf['mobility']['transition']['utility_xgboost']['p2d']['result'])

    for percentage_int in range(5, 100, 5):
        percentage = percentage_int / 100
        od_sample = od.sample(frac=percentage)
        demand = od_sample.groupby(['original_cube', 'destination_cube']).size().rename('sample')
        all_and_sample = pd.merge(od_pairs_demand, demand,
                                  how='left',
                                  left_on=['original_cube', 'destination_cube'], right_index=True).fillna(0.0001)
        kl_sample = all_and_sample.groupby("original_cube").apply(
            lambda x: kl(normalize(x["demand_17_et"].to_numpy().reshape(1, -1), norm='l1'),
                         normalize(x["sample"].to_numpy().reshape(1, -1), norm='l1'))
        )
        print(percentage, kl_sample.mean())
