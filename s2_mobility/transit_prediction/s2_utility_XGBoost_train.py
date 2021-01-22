import os
import yaml
import argparse
import pickle
import random
import xgboost
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import display


def mape_vectorized_v2(a, b):
    b = b.reshape(1, -1)
    mask = a != 0
    a = a[mask]
    b = b[mask]
    return (np.fabs(a - b) / a).mean()


def kl(P, Q):
    epsilon = 0.00001
    _P, _Q = P + epsilon, Q + epsilon
    divergence = np.sum(_P * np.log(_P / _Q))
    return divergence


def split_by_o(x, y, ratio=0.7):
    unique_o = y["original_cube"].unique().tolist()
    train_o = random.sample(unique_o, int(len(unique_o) * ratio))

    train_idx = y.loc[y["original_cube"].isin(train_o)].index.to_list()
    val_idx = y.loc[~y["original_cube"].isin(train_o)].index.to_list()
    train_x, train_y, val_x, val_y = x[train_idx], y.iloc[train_idx], x[val_idx], y.iloc[val_idx]
    return train_x, train_y, val_x, val_y


if __name__ == '__main__':
    display.configure_logging()
    display.configure_pandas()

    parser = argparse.ArgumentParser(description='Transition Prediction Utility (XGBoost) Train')
    parser.add_argument('--task', type=str, default='p2d', choices=['p2d', 'd2p'])
    args = parser.parse_args()
    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    # random.seed(10)
    split_by_o_flag = False

    p2d_x = np.load(conf["mobility"]["transition"]["utility_xgboost"][args.task]["train_feature"])
    p2d_gt = pd.read_csv(conf["mobility"]["transition"]["utility_xgboost"][args.task]["train_gt"])

    if split_by_o_flag:
        p2d_train_x, _p2d_train_y, p2d_val_x, _p2d_val_y = split_by_o(p2d_x, p2d_gt, ratio=0.2)
    else:
        p2d_train_x, p2d_val_x, _p2d_train_y, _p2d_val_y = train_test_split(p2d_x, p2d_gt, test_size=0.2)
    p2d_train_y, p2d_val_y = _p2d_train_y["rate"].to_numpy(), _p2d_val_y["rate"].to_numpy()
    print(p2d_train_x.shape, p2d_train_y.shape, p2d_val_x.shape, p2d_val_y.shape)

    scaler = StandardScaler()
    p2d_train_x = scaler.fit_transform(p2d_train_x)
    p2d_val_x = scaler.transform(p2d_val_x)

    xgbr = xgboost.XGBRegressor(verbosity=0, n_estimators=100, validate_parameters=2, learning_rate=0.05,
                                min_child_weight=5, max_depth=8)
    xgbr.fit(p2d_train_x, p2d_train_y)
    p2d_pred_y = xgbr.predict(p2d_val_x)

    print("MAE: {0:.4f}, MSE: {1:.4f}, RMSE: {2:.4f}, MAPE: {3:.4f}, R2: {4:.4f}".format(
        mean_absolute_error(p2d_val_y, p2d_pred_y), mean_squared_error(p2d_val_y, p2d_pred_y),
        mean_squared_error(p2d_val_y, p2d_pred_y, squared=False),
        mape_vectorized_v2(p2d_val_y.reshape(1, -1), p2d_pred_y), xgbr.score(p2d_val_x, p2d_val_y)
    ))

    # Further calculate transition probability distribution error
    p2d_val_y_for_kl = _p2d_val_y.copy()
    p2d_val_y_for_kl["pred_rate"] = p2d_pred_y
    kl = p2d_val_y_for_kl.groupby("original_cube").apply(
        lambda x: kl(normalize((x["pred_rate"]*x["demand_all"]).to_numpy().reshape(1, -1), axis=1),
                     normalize((x["demand_et"]).to_numpy().reshape(1, -1), axis=1))
    )
    print("Kullback–Leibler divergence: {:.4f}".format(kl.mean()))

    # Save fitted XGBRegressor model and StandardScaler to local.
    with open(conf["mobility"]["transition"]["utility_xgboost"][args.task]["model"], 'wb') as f:
        pickle.dump(xgbr, f)
    with open(conf["mobility"]["transition"]["utility_xgboost"][args.task]["scaler"], 'wb') as f:
        pickle.dump(scaler, f)
