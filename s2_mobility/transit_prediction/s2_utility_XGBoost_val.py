import os
import yaml
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import display
from s2_utility_XGBoost_train import kl

if __name__ == '__main__':
    display.configure_logging()
    display.configure_pandas()

    parser = argparse.ArgumentParser(description='Transition Prediction Utility (XGBoost) Validation')
    parser.add_argument('--task', type=str, default='p2d', choices=['p2d', 'd2p'])
    args = parser.parse_args()
    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    p2d_x = np.load(conf["mobility"]["transition"]["utility_xgboost"][args.task]["val_feature"])
    gt = pd.read_csv(conf["mobility"]["transition"]["utility_xgboost"][args.task]["val_gt"])

    # Load fitted XGBRegressor model and StandardScaler
    with open(conf["mobility"]["transition"]["utility_xgboost"][args.task]["model"], 'rb') as f:
        xgbr = pickle.load(f)
    with open(conf["mobility"]["transition"]["utility_xgboost"][args.task]["scaler"], 'rb') as f:
        scaler = pickle.load(f)

    p2d_pred_y = xgbr.predict(p2d_x)
    gt["pred_rate"] = p2d_pred_y

    kl_pred = gt.groupby("original_cube").apply(
        lambda x: kl(MinMaxScaler().fit_transform(x["demand_17_et"].to_numpy().reshape(-1, 1)),
                     MinMaxScaler().fit_transform((x["pred_rate"] * x["demand_all"]).to_numpy().reshape(-1, 1)))
    )

    kl_all = gt.groupby("original_cube").apply(
        lambda x: kl(MinMaxScaler().fit_transform(x["demand_17_et"].to_numpy().reshape(-1, 1)),
                     MinMaxScaler().fit_transform(x["demand_all"].to_numpy().reshape(-1, 1)))
    )

    kl_et = gt.groupby("original_cube").apply(
        lambda x: kl(MinMaxScaler().fit_transform(x["demand_17_et"].to_numpy().reshape(-1, 1)),
                     MinMaxScaler().fit_transform(x["demand_14_et"].to_numpy().reshape(-1, 1)))
    )
    print("Kullback–Leibler divergence\nprediction: {:.2f}, 14all: {:.2f}, 14EV: {:.2f}"
          .format(kl_pred.mean(), kl_all.mean(), kl_et.mean())
          )

    # Postprocessing: build transition probability matrix for task (p2d or d2p)
    gt['pred'] = gt['demand_all'] * gt['pred_rate']

    mat = gt.pivot(index='original_cube', columns='destination_cube', values='pred').fillna(0)
    gt.to_csv(conf['mobility']['transition']['utility_xgboost'][args.task]['result'], index=False)
    mat.to_csv(conf['mobility']['transition']['utility_xgboost'][args.task]['prob_mat_incomplete'])
