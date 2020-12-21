import os
import yaml
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import display
from utility_XGBoost_train import kl

if __name__ == '__main__':
    display.configure_logging()
    display.configure_pandas()

    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    p2d_x = np.load(conf["mobility"]["transition"]["utility_xgboost"]["p2d_val_feature"])
    p2d_gt = pd.read_csv(conf["mobility"]["transition"]["utility_xgboost"]["p2d_val_gt"])

    # Load fitted XGBRegressor model and StandardScaler
    with open(conf["mobility"]["transition"]["utility_xgboost"]["p2d_model"], 'rb') as f:
        xgbr = pickle.load(f)
    with open(conf["mobility"]["transition"]["utility_xgboost"]["p2d_scaler"], 'rb') as f:
        scaler = pickle.load(f)

    p2d_pred_y = xgbr.predict(p2d_x)
    p2d_gt["pred_rate"] = p2d_pred_y

    kl = p2d_gt.groupby("original_cube").apply(
        lambda x: kl(MinMaxScaler().fit_transform(x["demand_17_et"].to_numpy().reshape(-1, 1)),
                     MinMaxScaler().fit_transform((x["pred_rate"]*x["demand_all"]).to_numpy().reshape(-1, 1)))
    )

    print(kl.mean())
