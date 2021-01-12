import os
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr
from s2_mobility.transit_prediction.s2_utility_XGBoost_train import kl

if __name__ == '__main__':
    # configure the working directory to the project root path
    with open("../../config.yaml", "r", encoding="utf8") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    os.chdir(conf["project_path"])

    _17_et_pred = pd.read_csv(
        conf['mobility']['transition']['utility_xgboost']['p2d']['prob_mat_incomplete'], index_col=0)
    _od = pd.read_csv(conf["mobility"]["transition"]["utility_xgboost"]['p2d']["val_gt"])
    _17_et_gt = _od.pivot(index='original_cube', columns='destination_cube', values='demand_17_et').fillna(0)
    _14_et = _od.pivot(index='original_cube', columns='destination_cube', values='demand_14_et').fillna(0)
    _14_all = _od.pivot(index='original_cube', columns='destination_cube', values='demand_all').fillna(0)

    _17_et_pred = normalize(_17_et_pred, norm='l1')
    _17_et_gt = normalize(_17_et_gt, norm='l1')
    _14_et = normalize(_14_et, norm='l1')
    _14_all = normalize(_14_all, norm='l1')

    print("shape of 14 et mat: ", _14_et.shape)
    print("shape of 14 all taxis mat: ", _14_all.shape)
    print("shape of 17 et mat: ", _17_et_gt.shape)
    print("shape of 17 et prediction mat: ", _17_et_pred.shape)
    print(np.multiply(_17_et_gt, _17_et_pred).sum())
    print(np.multiply(_17_et_gt, _14_all).sum())
    print(pearsonr(_17_et_gt.reshape(-1), _17_et_pred.reshape(-1)))
    print(pearsonr(_17_et_gt.reshape(-1), _14_all.reshape(-1)))

    print(kl(_17_et_pred.reshape(-1), _17_et_gt.reshape(-1)))
    print(kl(_14_all.reshape(-1), _17_et_gt.reshape(-1)))
    print(kl(_14_et.reshape(-1), _17_et_gt.reshape(-1)))
