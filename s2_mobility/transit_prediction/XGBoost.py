import os
import yaml
import datetime
import xgboost
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, pairwise

from utils import data_loader, display, vector_haversine_distances as vec_hs_dis, od_utils
from s1_preprocessing.hotspot.hotpots_discovery_utils import generate_cube_index


def mape_vectorized_v2(a, b):
    b = b.reshape(1, -1)
    mask = a != 0
    a = a[mask]
    b = b[mask]
    return (np.fabs(a - b) / a).mean()


if __name__ == '__main__':
    display.configure_logging()
    display.configure_pandas()
    # earth radius(km)
    AVG_EARTH_RADIUS = 6371.0088

    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    # Load all transactions and statistically count the transition demand.
    df_od = data_loader.load_od(scale='full', common=False)
    od_utils.filter_in_bbox(df_od)  # filter the transactions within the bounding box.
    df_od = generate_cube_index(df_od, m=100, n=200)
    demand = df_od.groupby(['original_cube', 'destination_cube']).size().rename("demand").reset_index()
    demand = demand.loc[demand['demand'] > 10]
    demand.reset_index(drop=True, inplace=True)

    # Load ET transactions and statistically count the ET transition demand.
    df_et_od = data_loader.load_od(scale='full', common=True)
    od_utils.filter_in_bbox(df_et_od)  # filter the transactions within the bounding box.
    df_et_od = generate_cube_index(df_et_od, m=100, n=200)
    et_demand = df_et_od.groupby(['original_cube', 'destination_cube']).size().rename("demand").reset_index()

    df_demands = pd.merge(demand, et_demand, how='left', on=['original_cube', 'destination_cube'],
                          suffixes=('_all', '_et'))
    df_demands = df_demands.fillna(0)
    df_demands['rate'] = df_demands['demand_et'] / df_demands['demand_all']
    df_od['duration'] = (df_od['end_time'] - df_od['begin_time']).dt.total_seconds()
    df_od = df_od[['original_cube', 'destination_cube', 'original_log', 'original_lat', 'destination_log',
                   'destination_lat', 'duration']].groupby(['original_cube', 'destination_cube']).mean()
    df_od.reset_index(inplace=True)

    # df_od_pairs are final od set which have both duration and demand (>10) info
    df_od_pairs = pd.merge(df_demands[['original_cube', 'destination_cube']],
                           df_od[
                               ['original_cube', 'destination_cube', 'original_log', 'original_lat', 'destination_log',
                                'destination_lat', 'duration']],
                           left_on=['original_cube', 'destination_cube'],
                           right_on=['original_cube', 'destination_cube'])

    # FEATURE: dis. calculate distance between od locations
    od_dis = vec_hs_dis.haversine_np(df_od_pairs['original_log'], df_od_pairs['original_lat'],
                                     df_od_pairs['destination_log'], df_od_pairs['destination_lat'])
    df_od_dis = pd.DataFrame(od_dis)

    # Load CS (charging station) information.
    df_cs, _ = data_loader.load_cs(scale='part', date=datetime.datetime(2014, 7, 1))
    df_cs = df_cs.loc[~df_cs['cs_name'].isin(['LJDL', 'E04', 'BN0002', 'F11', 'S1', 'S2', 'F12', 'F13', 'F15'])]
    df_cs.reset_index(drop=True, inplace=True)
    original_capacity = np.repeat(df_cs['chg_points'].values.reshape(1, -1), df_od_pairs.shape[0], axis=0)

    # calculate distance between original/destination location and CS location
    original_locations = df_od_pairs[['original_lat', 'original_log']].to_numpy()
    destination_locations = df_od_pairs[['destination_lat', 'destination_log']].to_numpy()
    cs_location = df_cs[['lat', 'lng']].to_numpy()
    original_distances_to_cs = pairwise.haversine_distances(np.radians(original_locations),
                                                            np.radians(cs_location)) * AVG_EARTH_RADIUS
    destination_distances_to_cs = pairwise.haversine_distances(np.radians(destination_locations),
                                                               np.radians(cs_location)) * AVG_EARTH_RADIUS
    df_original_distances_to_cs = pd.DataFrame(original_distances_to_cs)
    df_destination_distances_to_cs = pd.DataFrame(destination_distances_to_cs)

    def evaluate(n=3):
        # FEATURE: distances of n nearest charging stations for transition origin.
        a = df_original_distances_to_cs.to_numpy(copy=True)
        a.sort(axis=1)
        a = pd.DataFrame(a, df_original_distances_to_cs.index, df_original_distances_to_cs.columns)
        a = a.iloc[:, :n]
        # FEATURE: capacity of n nearest charging stations for transition origin.
        o_dissorted_capacity = np.take_along_axis(original_capacity, df_original_distances_to_cs.values.argsort(axis=1),
                                                  axis=1)
        o_dissorted_capacity = pd.DataFrame(o_dissorted_capacity)
        o_dissorted_capacity = o_dissorted_capacity.iloc[:, :n]
        # FEATURE: distances of n nearest charging stations for transition destination.
        b = df_destination_distances_to_cs.values
        b.sort(axis=1)
        b = pd.DataFrame(b, df_destination_distances_to_cs.index, df_destination_distances_to_cs.columns)
        b = b.iloc[:, :n]
        # FEATURE: capacity of n nearest charging stations for transition destination.
        d_dissorted_capacity = np.take_along_axis(original_capacity,
                                                  df_destination_distances_to_cs.values.argsort(axis=1),
                                                  axis=1)
        d_dissorted_capacity = pd.DataFrame(d_dissorted_capacity)
        d_dissorted_capacity = d_dissorted_capacity.iloc[:, :n]

        # make complete data set.
        data_set = pd.concat([df_od_pairs['duration'], od_dis, a, o_dissorted_capacity, b, d_dissorted_capacity,
                              df_demands['demand_et'], df_demands['rate']], axis=1)
        data_set = data_set.loc[data_set['demand_et'] > 10]
        train_x, val_x, train_y, val_y = train_test_split(data_set.iloc[:, :-2].values,
                                                          data_set.iloc[:, -1].values, test_size=0.2)
        print('train data shape:', train_x.shape)
        print('test data shape:', val_x.shape)
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.transform(val_x)

        gbm = xgboost.XGBRegressor(verbosity=0, n_estimators=100, validate_parameters=2,
                                   learning_rate=0.05, min_child_weight=5, max_depth=8)

        gbm.fit(train_x, train_y)
        predict_y = gbm.predict(val_x)

        print('mae:', mean_absolute_error(val_y, predict_y),
              'mse:', mean_squared_error(val_y, predict_y),
              'rmse:', mean_squared_error(val_y, predict_y, squared=False))
        print('mape:', mape_vectorized_v2(val_y.reshape(1, -1), predict_y))
        print('R2:', gbm.score(val_x, val_y))
        # un comment following code segment to show cases of ground truth and prediction result.
        # print("ground truth:", val_y[30:36], )
        # print("prediction result:", predict_y[30:36])
        # print("ground truth:", val_y[50:56], )
        # print("prediction result:", predict_y[50:56])

    # search for the best neighbor number.
    for i in range(3, 4):
        print('Use {} nearest cs as feature:'.format(i))
        evaluate(n=i)
